#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4

#include <madrona/bvh.hpp>
#include <madrona/mesh_bvh3.hpp>
//#include "/home/warrenxia/Desktop/MadronaBVH/madrona_escape_room/external/madrona/src/mw/device/include/madrona/bvh.hpp"

using namespace madrona;

namespace sm {

// Only shared memory to be used
extern __shared__ uint8_t buffer[];

}

extern "C" __constant__ BVHParams bvhParams;

//BVHParams bvhParams;

constexpr int RAYCAST_WIDTH = 64;
constexpr int RAYCAST_HEIGHT = 64;

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
            bool hit = bvh->traceRay(ray_start, ray_dir, &t, &normal,0);
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
