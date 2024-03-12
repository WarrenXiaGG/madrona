#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>

namespace madrona::render {

// This will be attached to any entity that wants to be a viewer
struct RenderCamera {
    Entity cameraEntity;

    // 1.0 / tanf(fovy * 0.5)
    float fovScale;
    float zNear;

    math::Vector3 cameraOffset;
};

// This will be attached to any renderable entity
struct Renderable {
    Entity renderEntity;
};




// TODO: Make sure to move this to private headers which can be
// included by the device code for the BVH.

// For private usage - not to be used by user.
using MortonCode = uint32_t;

// For private usage - not to be used by user.
struct alignas(16) PerspectiveCameraData {
    math::Vector3 position;
    math::Quat rotation;
    float xScale;
    float yScale;
    float zNear;
    int32_t worldIDX;
    uint32_t pad;
};

// For private usage - not to be used by user.
struct alignas(16) InstanceData {
    math::Vector3 position;
    math::Quat rotation;
    math::Diag3x3 scale;
    int32_t objectID;
    int32_t worldIDX;
};

struct RenderOutput {
    uint8_t output[64][64][3];
};

struct BVHModel {
    void* ptr;
};

// Top level acceleration structure node
struct TLBVHNode {
    math::AABB aabb;
};

// For private usage - not to be used by user.
struct RenderableArchetype : public Archetype<
    InstanceData,

    // For BVH support, we need to sort these not just be world ID,
    // but first by morton code too.
    MortonCode,
    BVHModel,

    TLBVHNode
> {};

// For private usage - not to be used by user.
struct RenderCameraArchetype : public Archetype<
    PerspectiveCameraData,
    RenderOutput
> {};



struct RenderECSBridge;

namespace RenderingSystem {
    void registerTypes(ECSRegistry &registry,
                       const RenderECSBridge *bridge);

    TaskGraphNodeID setupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

    void init(Context &ctx,
              const RenderECSBridge *bridge);

    uint32_t * getVoxelPtr(Context &ctx);

    void makeEntityRenderable(Context &ctx,
                              Entity e);

    void attachEntityToView(Context &ctx,
                            Entity e,
                            float vfov_degrees,
                            float z_near,
                            const math::Vector3 &camera_offset);

    // Need to call these before destroying entities
    void cleanupViewingEntity(Context &ctx,
                              Entity e);
    void cleanupRenderableEntity(Context &ctx,
                                 Entity e);
};

}
