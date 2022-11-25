#include <madrona/physics.hpp>
#include <madrona/context.hpp>

namespace madrona {
using namespace base;
using namespace math;

namespace phys {

namespace broadphase {

BVH::BVH(CountT max_leaves)
    : nodes_((Node *)malloc(sizeof(Node) *
                            numInternalNodes(max_leaves))),
      num_nodes_(0),
      num_allocated_nodes_(numInternalNodes(max_leaves)),
      leaf_aabbs_((LeafAABB *)malloc(sizeof(LeafAABB) * max_leaves)),
      leaf_centers_((LeafCenter *)malloc(sizeof(LeafCenter) * max_leaves)),
      leaf_parents_((uint32_t *)malloc(sizeof(uint32_t) * max_leaves)),
      sorted_leaves_((int32_t *)malloc(sizeof(int32_t) * max_leaves)),
      num_leaves_(0),
      num_allocated_leaves_(max_leaves)
{}

CountT BVH::numInternalNodes(CountT num_leaves) const
{
    return utils::divideRoundUp(num_leaves - 1, CountT(3));
}

void BVH::rebuild()
{
    int32_t num_internal_nodes = numInternalNodes(num_leaves_);
    num_nodes_ = num_internal_nodes;
    assert(num_nodes_ <= num_allocated_nodes_);

    struct StackEntry {
        int32_t nodeID;
        int32_t parentID;
        int32_t offset;
        int32_t numObjs;
    };

    StackEntry stack[128];
    stack[0] = StackEntry {
        sentinel_,
        sentinel_,
        0,
        int32_t(num_leaves_),
    };

    int32_t cur_node_offset = 0;
    CountT stack_size = 1;

    while (stack_size > 0) {
        StackEntry &entry = stack[stack_size - 1];
        int32_t node_id;
        if (entry.numObjs <= 4) {
            node_id = cur_node_offset++;
            Node &node = nodes_[node_id];
            node.parentID = entry.parentID;

            for (int i = 0; i < 4; i++) {
                if (i < entry.numObjs) {
                    int32_t leaf_id = sorted_leaves_[entry.offset + i];
                    const auto &aabb = leaf_aabbs_[leaf_id];
                    leaf_parents_[leaf_id] = ((uint32_t)node_id << 2) | (uint32_t)i;

                    node.setLeaf(i, leaf_id);
                    node.minX[i] = aabb.pMin.x;
                    node.minY[i] = aabb.pMin.y;
                    node.minZ[i] = aabb.pMin.z;
                    node.maxX[i] = aabb.pMax.x;
                    node.maxY[i] = aabb.pMax.y;
                    node.maxZ[i] = aabb.pMax.z;
                } else {
                    node.children[i] = sentinel_;
                    node.minX[i] = FLT_MAX;
                    node.minY[i] = FLT_MAX;
                    node.minZ[i] = FLT_MAX;
                    node.maxX[i] = FLT_MIN;
                    node.maxY[i] = FLT_MIN;
                    node.maxZ[i] = FLT_MIN;
                }
            }
        } else if (entry.nodeID == sentinel_) {
            node_id = cur_node_offset++;
            // Record the node id in the stack entry for when this entry
            // is reprocessed
            entry.nodeID = node_id;

            Node &node = nodes_[node_id];
            for (int i = 0; i < 4; i++) {
                node.children[i] = sentinel_;
            }
            node.parentID = entry.parentID;

            // midpoint sort items
            auto midpoint_split = [this](
                    int32_t base, int32_t num_elems) {

                auto get_center = [this, base](int32_t offset) {
                    return leaf_centers_[sorted_leaves_[base + offset]];
                };

                Vector3 center_min {
                    FLT_MAX,
                    FLT_MAX,
                    FLT_MAX,
                };

                Vector3 center_max {
                    FLT_MIN,
                    FLT_MIN,
                    FLT_MIN,
                };

                for (int i = 0; i < num_elems; i++) {
                    const LeafCenter &center = get_center(i);
                    center_min = Vector3::min(center_min, center);
                    center_max = Vector3::max(center_max, center);
                }

                auto split = [&](auto get_component) {
                    float split_val = 0.5f * (get_component(center_min) +
                                              get_component(center_max));

                    int start = 0;
                    int end = num_elems;

                    while (start < end) {
                        while (start < end &&
                               get_component(get_center(start)) < split_val) {
                            ++start;
                        }

                        while (start < end && get_component(
                                get_center(end - 1)) >= split_val) {
                            --end;
                        }

                        if (start < end) {
                            std::swap(sorted_leaves_[base + start],
                                      sorted_leaves_[base + end - 1]);
                            ++start;
                            --end;
                        }
                    }

                    if (start > 0 && start < num_elems) {
                        return start;
                    } else {
                        return num_elems / 2;
                    }
                };

                Vector3 center_diff = center_max - center_min;
                if (center_diff.x > center_diff.y &&
                    center_diff.x > center_diff.z) {
                    return split([](Vector3 v) {
                        return v.x;
                    });
                } else if (center_diff.y > center_diff.x &&
                           center_diff.y > center_diff.z) {
                    return split([](Vector3 v) {
                        return v.y;
                    });
                } else {
                    return split([](Vector3 v) {
                        return v.z;
                    });
                }
            };

            int32_t second_split = midpoint_split(entry.offset, entry.numObjs);
            int32_t num_h1 = second_split;
            int32_t num_h2 = entry.numObjs - second_split;

            int32_t first_split = midpoint_split(entry.offset, num_h1);
            int32_t third_split =
                midpoint_split(entry.offset + second_split, num_h2);

#if 0
            printf("%u %u\n", entry.offset, entry.numObjs);
            printf("[%u %u) [%u %u) [%u %u) [%u %u)\n",
                   entry.offset, entry.offset + first_split,
                   entry.offset + first_split, entry.offset + first_split + num_h1 - first_split,
                   entry.offset + num_h1, entry.offset + num_h1 + third_split,
                   entry.offset + num_h1 + third_split, entry.offset + num_h1 + third_split + num_h2 - third_split);
#endif

            // Setup stack to recurse into fourths. Put fourths on stack in
            // reverse order to preserve left-right depth first ordering

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + num_h1 + third_split,
                num_h2 - third_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + num_h1,
                third_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + first_split,
                num_h1 - first_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset,
                first_split,
            };

            // Don't finish processing this node until children are processed
            continue;
        } else {
            // Revisiting this node after having processed children
            node_id = entry.nodeID;
        }

        // At this point, remove the current entry from the stack
        stack_size -= 1;

        Node &node = nodes_[node_id];
        if (node.parentID == -1) {
            continue;
        }

        AABB combined_aabb  = AABB::invalid(); 
        for (int i = 0; i < 4; i++) {
            if (node.children[i] == sentinel_) {
                break;
            }

            combined_aabb = AABB::merge(combined_aabb, AABB {
                .pMin = {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                .pMax = {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
                },
            });
        }

        Node &parent = nodes_[node.parentID];
        int child_offset;
        for (child_offset = 0; ; child_offset++) {
            if (parent.children[child_offset] == sentinel_) {
                break;
            }
        }

        parent.children[child_offset] = entry.nodeID;
        parent.minX[child_offset] = combined_aabb.pMin.x;
        parent.minY[child_offset] = combined_aabb.pMin.y;
        parent.minZ[child_offset] = combined_aabb.pMin.z;
        parent.maxX[child_offset] = combined_aabb.pMax.x;
        parent.maxY[child_offset] = combined_aabb.pMax.y;
        parent.maxZ[child_offset] = combined_aabb.pMax.z;
    }
}

void BVH::refit(LeafID *moved_leaf_ids, CountT num_moved)
{
    for (CountT i = 0; i < num_moved; i++) {
        int32_t leaf_id = moved_leaf_ids[i].id;
        const LeafAABB &leaf_aabb = leaf_aabbs_[leaf_id];
        uint32_t leaf_parent = leaf_parents_[leaf_id];

        int32_t node_idx = int32_t(leaf_parent >> 2_u32);
        int32_t sub_idx = int32_t(leaf_parent & 3);

        Node &leaf_node = nodes_[node_idx];
        leaf_node.minX[sub_idx] = leaf_aabb.pMin.x;
        leaf_node.minY[sub_idx] = leaf_aabb.pMin.y;
        leaf_node.minZ[sub_idx] = leaf_aabb.pMin.z;
        leaf_node.maxX[sub_idx] = leaf_aabb.pMax.x;
        leaf_node.maxY[sub_idx] = leaf_aabb.pMax.y;
        leaf_node.maxZ[sub_idx] = leaf_aabb.pMax.z;

        int32_t child_idx = node_idx;
        node_idx = leaf_node.parentID;

        while (node_idx != sentinel_) {
            Node &node = nodes_[node_idx];
            int child_offset = -1;
            for (int j = 0; j < 4; j++) {
                if (node.children[j] == child_idx) {
                    child_offset = j;
                    break;
                }
            }
            assert(child_offset != -1);

            bool expanded = false;
            if (leaf_aabb.pMin.x < node.minX[child_offset]) {
                node.minX[child_offset] = leaf_aabb.pMin.x;
                expanded = true;
            }

            if (leaf_aabb.pMin.y < node.minY[child_offset]) {
                node.minY[child_offset] = leaf_aabb.pMin.y;
                expanded = true;
            }

            if (leaf_aabb.pMin.z < node.minZ[child_offset]) {
                node.minZ[child_offset] = leaf_aabb.pMin.z;
                expanded = true;
            }

            if (leaf_aabb.pMax.x > node.maxX[child_offset]) {
                node.maxX[child_offset] = leaf_aabb.pMax.x;
                expanded = true;
            }

            if (leaf_aabb.pMax.y > node.maxY[child_offset]) {
                node.maxY[child_offset] = leaf_aabb.pMax.y;
                expanded = true;
            }

            if (leaf_aabb.pMax.z > node.maxZ[child_offset]) {
                node.maxZ[child_offset] = leaf_aabb.pMax.z;
                expanded = true;
            }

            if (!expanded) {
                break;
            }
            
            child_idx = node_idx;
            node_idx = node.parentID;
        }
    }
}

void BVH::updateLeaf(const Position &position,
                     const Rotation &rotation,
                     const LeafID &leaf_id)
{
    Mat3x4 model_mat = Mat3x4::fromTRS(position, rotation);

    // No actual mesh, just hardcode a fake 2 *unit cube centered around
    Vector3 cube[8] = {
        model_mat.txfmPoint(Vector3 {-1.f, -1.f, -1.f}),
        model_mat.txfmPoint(Vector3 { 1.f, -1.f, -1.f}),
        model_mat.txfmPoint(Vector3 { 1.f,  1.f, -1.f}),
        model_mat.txfmPoint(Vector3 {-1.f,  1.f, -1.f}),
        model_mat.txfmPoint(Vector3 {-1.f, -1.f,  1.f}),
        model_mat.txfmPoint(Vector3 { 1.f, -1.f,  1.f}),
        model_mat.txfmPoint(Vector3 { 1.f,  1.f,  1.f}),
        model_mat.txfmPoint(Vector3 {-1.f,  1.f,  1.f}),
    };

    LeafAABB &leaf_aabb = leaf_aabbs_[leaf_id.id];
    LeafCenter &leaf_center = leaf_centers_[leaf_id.id];

    leaf_aabb = AABB::point(cube[0]);
    for (int i = 1; i < 8; i++) {
        leaf_aabb.expand(cube[i]);
    }

    leaf_center = (leaf_aabb.pMin + leaf_aabb.pMax) / 2;
    sorted_leaves_[leaf_id.id] = leaf_id.id;

    printf("%d (%f %f %f)\n", leaf_id.id, leaf_center.x, leaf_center.y, leaf_center.z);
}

void BVH::updateLeavesSystem(
    Context &ctx,
    Loc sys_loc,
    const Position &pos,
    const Rotation &rot,
    const LeafID &leaf_id)
{
    BVH &bvh = ctx.getUnsafe<BVH>(sys_loc);
    bvh.updateLeaf(pos, rot, leaf_id);
}

void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<LeafID>();
    registry.registerComponent<BVH>();
}

TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                             Span<const TaskGraph::NodeID> deps)
{
    auto preprocess_node = builder.parallelForNode<Context,
        BVH::updateLeavesSystem, Position, Rotation, LeafAABB, LeafCenter>(deps);

    return preprocess_node;
}


}

}
}