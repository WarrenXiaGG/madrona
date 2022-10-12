/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/custom_context.hpp>
#include <madrona/math.hpp>

#include <random>

namespace fvs {

// Components
struct Position : madrona::math::Vector3 {
    Position(Vector3 o)
        : Vector3(o)
    {}
};

struct alignas(MADRONA_CACHE_LINE) Health {
    std::atomic_int hp;
};

struct Mana {
    float mp;
};

struct Quiver {
    int numArrows;
};

struct Action {
    float remainingTime;
};

struct CleanupEntity : madrona::Entity {
    CleanupEntity(madrona::Entity e)
        : Entity(e)
    {}
};

// Archetypes
struct Dragon : madrona::Archetype<Position, Health, Action, Mana> {};
struct Knight : madrona::Archetype<Position, Health, Action, Quiver> {};
struct CleanupTracker : madrona::Archetype<CleanupEntity> {};

class Engine;

struct BenchmarkConfig {
    bool enable;
    uint32_t numTicks;
    uint32_t numKnights;
    uint32_t numDragons;
};

// Per-world state object (one per-world created by JobManager)
struct Game : public madrona::WorldBase {
    static void entry(Engine &ctx, const BenchmarkConfig &bench);

    Game(Engine &ctx, const BenchmarkConfig &bench);
    void tick(Engine &ctx);
    void gameLoop(Engine &ctx);

    void benchmarkTick(Engine &ctx);
    void benchmark(Engine &ctx, const BenchmarkConfig &bench);

    uint64_t tickCount;

    float deltaT;
    float moveSpeed;
    float manaRegenRate;
    float castTime;
    float shootTime;

    madrona::math::AABB worldBounds;

    madrona::Query<Position, Action> actionQuery;
    madrona::Query<Position, Health> healthQuery;
    madrona::Query<Action, Mana> casterQuery;
    madrona::Query<Action, Quiver> archerQuery;
    madrona::Query<madrona::Entity, Health> cleanupQuery;
};

// madrona::Context subclass, allows easy access to per-world state through
// game() method
class Engine : public::madrona::CustomContext<Engine, Game> {
public:
    using CustomContext::CustomContext;
    inline Game & game() { return data(); }
};

}
