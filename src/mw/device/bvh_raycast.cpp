#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4

#include <madrona/bvh.hpp>
#include <madrona/mesh_bvh.hpp>

using namespace madrona;

namespace sm {

// Only shared memory to be used
extern __shared__ uint8_t buffer[];

}

extern "C" __constant__ BVHParams bvhParams;

//BVHParams bvhParams;

constexpr int RAYCAST_WIDTH = 64;
constexpr int RAYCAST_HEIGHT = 64;

// Trace a ray through the top level structure.
static __device__ bool traceRayTLAS(uint32_t world_idx,
                                    uint32_t view_idx,
                                    uint32_t pixel_x, uint32_t pixel_y,
                                    const math::Vector3 &ray_o,
                                    math::Vector3 ray_d,
                                    float *out_hit_t,
                                    math::Vector3 *out_hit_normal,
                                    float t_max)
{
#define INSPECT(...) if (pixel_x == 32 && pixel_y == 32) { printf(__VA_ARGS__); }

    static constexpr float epsilon = 0.00001f;

    if (ray_d.x == 0.f) {
        ray_d.x += epsilon;
    }

    if (ray_d.y == 0.f) {
        ray_d.y += epsilon;
    }

    if (ray_d.z == 0.f) {
        ray_d.z += epsilon;
    }

    using namespace madrona::math;

    BVHInternalData *internal_data = bvhParams.internalData;

    // internal_nodes_offset contains the offset to instance attached
    // data of world being operated on.
    uint32_t internal_nodes_offset = bvhParams.instanceOffsets[world_idx];

    LBVHNode *internal_nodes = internal_data->internalNodes + internal_nodes_offset;
    LBVHNode *leaves = internal_data->leaves + internal_nodes_offset;
    render::InstanceData *instances = bvhParams.instances + internal_nodes_offset;

    render::BVHModel *bvh_models = bvhParams.bvhModels + internal_nodes_offset;

    phys::TraversalStack stack = {};

    // This starts us at root
    stack.push(1);

    bool ray_hit = false;
    Vector3 closest_hit_normal = {};

    while (stack.size > 0) {
        // Can be negative if this is a leaf node
        int32_t node_store_idx = stack.pop();

#if 1
        bool is_leaf = false;
        int32_t node_idx = LBVHNode::storeIdxToChildIdx(node_store_idx, is_leaf);

        assert(!is_leaf);

        LBVHNode *node = &internal_nodes[node_idx];

        int32_t children_indices[] = {
            node->left, node->right
        };

        // The tree is currently 2 wide
        for (CountT i = 0; i < 2; ++i) {
            if (children_indices[i] == 0) {
                // No child in this location
                continue;
            }

            bool child_is_leaf = false;
            int32_t child_node_idx = 
                LBVHNode::storeIdxToChildIdx(children_indices[i], child_is_leaf);

            LBVHNode *child_node = child_is_leaf ? &leaves[child_node_idx] :
                &internal_nodes[child_node_idx];

            math::AABB child_aabb = child_node->aabb;

            float aabb_hit_t, aabb_far_t;
            Diag3x3 inv_ray_d = { 1.f/ray_d.x, 1.f/ray_d.y, 1.f/ray_d.z };
            bool intersect_aabb = child_aabb.rayIntersects(ray_o, inv_ray_d,
                    4.0f, t_max, aabb_hit_t, aabb_far_t);

            if (aabb_hit_t <= t_max) {
#if 0
                INSPECT("(%d %d) Hit an AABB (%d)\n", pixel_x, pixel_y, 
                        children_indices[i]);
#endif

                // If the near T of the box intersection happens before the closest
                // intersection we got thus far, try tracing through.

                if (child_is_leaf) {
#if 1

                    // Child node idx is the index of the mesh bvh
                    // LBVHNode *leaf_node = &leaves[child_node_idx];

                    render::BVHModel *model = &bvh_models[child_node_idx];
                    render::InstanceData *instance_data = &instances[child_node_idx];
                    phys::MeshBVH *model_bvh = (phys::MeshBVH *)model->bvh;

                    // Now we trace through this model.
                    float hit_t;
                    Vector3 leaf_hit_normal;

                    // Ok we are going to just do something stupid.
                    //
                    // Also need to bound the mesh bvh trace ray by t_max.

                    phys::MeshBVH::AABBTransform txfm = {
                        instance_data->position,
                        instance_data->rotation,
                        instance_data->scale
                    };

                    Vector3 txfm_ray_o = instance_data->scale.inv() *
                        instance_data->rotation.inv().rotateVec(
                            (ray_o - instance_data->position));

                    Vector3 txfm_ray_d = instance_data->scale.inv() *
                        instance_data->rotation.inv().rotateVec(ray_d);

                    float t_scale = txfm_ray_d.length();

                    txfm_ray_d /= t_scale;

                    INSPECT("bvh vertices at : %p\n", model_bvh->vertices);

                    bool leaf_hit = model_bvh->traceRay(
                            txfm_ray_o, txfm_ray_d, &hit_t,
                            &leaf_hit_normal, &stack, txfm, t_max);

                    if (leaf_hit) {
                        // INSPECT("hit\n");

                        ray_hit = true;
                        t_max = hit_t * t_scale;

                        closest_hit_normal = instance_data->rotation.rotateVec(
                                instance_data->scale * leaf_hit_normal);
                        closest_hit_normal = closest_hit_normal.normalize();
                    }
#endif
                } else {
                    assert(stack.size < phys::TraversalStack::stackSize);

                    stack.push(children_indices[i]);
                }
            } else {
                // printf("ray failed intersection\n");
            }
        }
#endif
    }

    *out_hit_normal = closest_hit_normal;

    return ray_hit;
}

extern "C" __global__ void bvhRaycastEntry()
{
    const uint32_t num_worlds = bvhParams.numWorlds;
    const uint32_t total_num_views = bvhParams.viewOffsets[num_worlds-1] +
                                     bvhParams.viewCounts[num_worlds-1];

    // This is the number of views currently being processed.
    const uint32_t num_resident_views = gridDim.x;

    // This is the offset into the resident view processors that we are
    // currently in.
    const uint32_t resident_view_offset = blockIdx.x;

    uint32_t current_view_offset = resident_view_offset;

    while (current_view_offset < total_num_views) {
        // While we still have views to generate, trace.
        render::PerspectiveCameraData *view_data = 
            &bvhParams.views[current_view_offset];

        uint32_t world_idx = (uint32_t)view_data->worldIDX;


        // Initialize the ray and trace.
        math::Quat rot = view_data->rotation;
        math::Vector3 ray_start = view_data->position;
        math::Vector3 look_at = rot.inv().rotateVec({0, 1, 0});

        constexpr float theta = 1.5708f;

        const float h = tanf(theta / 2);

        const auto viewport_height = 2 * h;
        const auto viewport_width = viewport_height;
        const auto forward = look_at.normalize();

        auto u = rot.inv().rotateVec({1, 0, 0});
        auto v = cross(forward, u).normalize();

        auto horizontal = u * viewport_width;
        auto vertical = v * viewport_height;

        auto lower_left_corner = ray_start - horizontal / 2 - vertical / 2 + forward;

        uint32_t pixel_x = blockIdx.y * 16 + threadIdx.x;
        uint32_t pixel_y = blockIdx.z * 16 + threadIdx.y;

        float pixel_u = ((float)pixel_x) / (float)RAYCAST_WIDTH;
        float pixel_v = ((float)pixel_y) / (float)RAYCAST_HEIGHT;

        math::Vector3 ray_dir = lower_left_corner + pixel_u * horizontal + 
            pixel_v * vertical - ray_start;
        ray_dir = ray_dir.normalize();

        float t;
        math::Vector3 normal = {pixel_u * 2 - 1, pixel_v * 2 - 1, 0};
        normal = ray_dir;

        // For now, just hack in a t_max of 10000.
        bool hit = traceRayTLAS(
                world_idx, current_view_offset, 
                pixel_x, pixel_y,
                ray_start, ray_dir, 
                &t, &normal, 10000.f);

        if (hit) {
            bvhParams.renderOutput[current_view_offset].output[pixel_x][pixel_y][0] =
                (normal.x * 0.5f + 0.5f) * 255;
            bvhParams.renderOutput[current_view_offset].output[pixel_x][pixel_y][1] =
                (normal.y * 0.5f + 0.5f) * 255;
            bvhParams.renderOutput[current_view_offset].output[pixel_x][pixel_y][2] =
                (normal.z * 0.5f + 0.5f) * 255;
        } else {
            bvhParams.renderOutput[current_view_offset].output[pixel_x][pixel_y][0] = 0;
            bvhParams.renderOutput[current_view_offset].output[pixel_x][pixel_y][1] = 0;
            bvhParams.renderOutput[current_view_offset].output[pixel_x][pixel_y][2] = 0;
        }

        current_view_offset += num_resident_views;


        __syncthreads();
    }
}

#if 0
extern "C" __global__ void bvhRaycastEntry()
{
    const int32_t num_views = bvhParams.numWorlds * 2;
    const int32_t num_views_per_grid = std::ceil(num_views / (float)gridDim.x);
    int objectRenderIndex = 0;
    phys::MeshBVH2* bvh = (phys::MeshBVH2*)bvhParams.bvhModels[objectRenderIndex].ptr;

    for(int view_i = 0; view_i < num_views_per_grid; view_i++) {
        int view = blockIdx.x * num_views_per_grid + view_i;
        if(view >= num_views){
            return;
        }
        int viewIndex = bvhParams.viewOffsets[view];
        int32_t idx = threadIdx.x;

        math::Quat rot = bvhParams.views[view].rotation;
        math::Vector3 ray_start = bvhParams.views[view].position;
        math::Vector3 look_at = rot.inv().rotateVec({0, 1, 0});

        constexpr float theta = 1.5708f;
        const float h = tanf(theta / 2);
        const auto viewport_height = 2 * h;
        const auto viewport_width = viewport_height;
        const auto forward = look_at.normalize();
        auto u = rot.inv().rotateVec({1, 0, 0});
        auto v = cross(forward, u).normalize();
        auto horizontal = u * viewport_width;
        auto vertical = v * viewport_height;
        auto lower_left_corner = ray_start - horizontal / 2 - vertical / 2 + forward;
        auto traceRay = [&](int32_t instanceIDX, int32_t idx, int32_t idy, int32_t subthread) {
            int pixelY = idy;
            int pixelX = idx;
            float v = ((float) pixelY) / RAYCAST_WIDTH;
            float u = ((float) pixelX) / RAYCAST_HEIGHT;

            math::Vector3 ray_dir = lower_left_corner + u * horizontal + v * vertical - ray_start;
            ray_dir = ray_dir.normalize();

            float t;
            math::Vector3 normal = {u * 2 - 1, v * 2 - 1, 0};
            normal = ray_dir;

            // Temporay:
            phys::TraversalStack stack;
            stack.size = 0;

            bool hit = bvh->traceRay(ray_start, ray_dir, &t, &normal,
                                     0, &stack);

            if (hit && subthread == 0) {
                bvhParams.renderOutput[instanceIDX].output[pixelX][pixelY][0] = (normal.x * 0.5f + 0.5f) * 255;
                bvhParams.renderOutput[instanceIDX].output[pixelX][pixelY][1] = (normal.y * 0.5f + 0.5f) * 255;
                bvhParams.renderOutput[instanceIDX].output[pixelX][pixelY][2] = (normal.z * 0.5f + 0.5f) * 255;
            } else if (subthread == 0) {
                bvhParams.renderOutput[instanceIDX].output[pixelX][pixelY][0] = 0;
                bvhParams.renderOutput[instanceIDX].output[pixelX][pixelY][1] = 0;
                bvhParams.renderOutput[instanceIDX].output[pixelX][pixelY][2] = 0;
            }
        };
        //printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
//gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

        traceRay(view, blockIdx.y * 16 + threadIdx.x, blockIdx.z * 16 + threadIdx.y, 0);
    }
}
#endif
