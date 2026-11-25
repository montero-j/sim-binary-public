/*! \file jam_output.cpp
 * \brief Output of each clog: contacts and stress tensor of each grain.
 *
 * See jam_output.hpp for the sign convention.
 */

#include "jam_output.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace {

/*! frames_<dirID>/<prefix><5-digit jam id><ext> */
std::string jam_path(const GlobalSetup *gs, const std::string &prefix,
                     int jamIndex, const std::string &ext) {
    std::ostringstream ss;
    ss << "frames_" << gs->dirID << "/" << prefix << std::setfill('0')
       << std::setw(5) << jamIndex << ext;
    return ss.str();
}

const BodyData *body_data(const b2Body *b) {
    return reinterpret_cast<const BodyData *>(b->GetUserData().pointer);
}

/*! true for the grains written to atasco_<jam>.xy */
bool is_active_grain(const b2Body *b) {
    const BodyData *bd = body_data(b);
    return bd && bd->isGrain && bd->isIn;
}

/*! Area of the grain in Box2D: the one that, times the density, gives its
 * mass. For polygons, the area of the vertices (without the skin). */
double body_area(const b2Body *b) {
    double area = 0.0;
    for (const b2Fixture *f = b->GetFixtureList(); f; f = f->GetNext()) {
        b2MassData md;
        f->GetShape()->ComputeMass(&md, 1.0f);
        area += md.mass;
    }
    return area;
}

/*! Per-grain sums. */
struct GrainAcc {
    const b2Body *body = nullptr;
    double area = 0.0;
    double s[4] = {0.0, 0.0, 0.0, 0.0};  // sum F_i l_j: xx, xy, yx, yy
    double sn[4] = {0.0, 0.0, 0.0, 0.0}; // same, normal force only
    double F[2] = {0.0, 0.0};            // net contact force
    double tau = 0.0;                    // net contact torque (l x F)
    int z = 0;   // contacts with total normal impulse > 0
    int z_w = 0; // ... of which with walls
    int z_f = 0; // ... of which with two active points (face-face)
};

} // namespace

void saveAtascoContacts(b2World *world, const GlobalSetup *gs, int jamIndex) {
    const std::string fileName = jam_path(gs, "atasco_contacts_", jamIndex, ".dat");
    std::ofstream f(fileName.c_str());
    if (!f.is_open()) {
        std::cout << "ERROR: could not create the clog contact file."
                  << std::endl;
        return;
    }
    f << "# ID_A ID_B PosX PosY NormalImpulse TangentImpulse\n";
    f << std::scientific << std::setprecision(6);
    for (b2Contact *c = world->GetContactList(); c; c = c->GetNext()) {
        if (!c->IsTouching()) continue;
        const BodyData *bdA = body_data(c->GetFixtureA()->GetBody());
        const BodyData *bdB = body_data(c->GetFixtureB()->GetBody());
        if (!bdA || !bdB) continue;
        b2WorldManifold wm;
        c->GetWorldManifold(&wm);
        const b2Manifold *m = c->GetManifold();
        for (int i = 0; i < m->pointCount; ++i) {
            f << bdA->gID << " " << bdB->gID << " " << wm.points[i].x << " "
              << wm.points[i].y << " " << m->points[i].normalImpulse << " "
              << m->points[i].tangentImpulse << "\n";
        }
    }
    f.close();
    std::cout << "# Clog CONTACTS snapshot saved: " << fileName
              << std::endl;
}

void saveAtascoTensor(b2World *world, const GlobalSetup *gs, int jamIndex,
                      double tSim) {
    const double dt = gs->tStep;
    const double inv_dt = 1.0 / dt;

    // Active grains, sorted by gID
    std::map<int, GrainAcc> grains;
    for (b2Body *b = world->GetBodyList(); b; b = b->GetNext()) {
        if (b->GetType() != b2_dynamicBody || !is_active_grain(b)) continue;
        GrainAcc g;
        g.body = b;
        g.area = body_area(b);
        grains[body_data(b)->gID] = g;
    }

    const std::string fullName =
        jam_path(gs, "atasco_contacts_full_", jamIndex, ".dat");
    std::ofstream fc(fullName.c_str());
    if (!fc.is_open()) {
        std::cout << "ERROR: could not create " << fullName << std::endl;
        return;
    }
    fc << "# t: " << std::setprecision(10) << tSim << " dt: " << dt
       << " atasco: " << jamIndex << "\n"
       << "# One row per manifold point. n: Box2D normal, from A to B."
          " Fn, Ft: impulses / dt.\n"
       << "# Force on B = Fn (nx, ny) + Ft (ny, -nx); on A, the "
          "opposite. (xA, yA), (xB, yB): centers of mass (nan for walls).\n"
       << "# mtype: 0 circles, 1 faceA (reference face on A), 2 faceB."
          " i: point index, npc: points in the manifold.\n"
       << "# ID_A ID_B i npc mtype PosX PosY nx ny Fn Ft xA yA xB yB\n";
    fc << std::scientific << std::setprecision(9);

    for (b2Contact *c = world->GetContactList(); c; c = c->GetNext()) {
        if (!c->IsTouching()) continue;
        b2Body *bA = c->GetFixtureA()->GetBody();
        b2Body *bB = c->GetFixtureB()->GetBody();
        const BodyData *bdA = body_data(bA);
        const BodyData *bdB = body_data(bB);
        if (!bdA || !bdB) continue;
        b2WorldManifold wm;
        c->GetWorldManifold(&wm);
        const b2Manifold *m = c->GetManifold();
        const b2Vec2 n = wm.normal;
        const b2Vec2 t = b2Cross(n, 1.0f); // (n.y, -n.x), as in b2ContactSolver
        const bool gA = bdA->isGrain, gB = bdB->isGrain;
        const b2Vec2 cA = bA->GetWorldCenter();
        const b2Vec2 cB = bB->GetWorldCenter();

        double fn_sum = 0.0;
        int n_active = 0;
        for (int i = 0; i < m->pointCount; ++i) {
            const double fn = m->points[i].normalImpulse * inv_dt;
            const double ft = m->points[i].tangentImpulse * inv_dt;
            fn_sum += fn;
            if (fn > 0.0) ++n_active;
            const b2Vec2 p = wm.points[i];

            fc << bdA->gID << " " << bdB->gID << " " << i << " "
               << m->pointCount << " " << static_cast<int>(m->type) << " "
               << p.x << " " << p.y << " " << n.x << " " << n.y << " " << fn
               << " " << ft << " ";
            if (gA) fc << cA.x << " " << cA.y << " ";
            else fc << "nan nan ";
            if (gB) fc << cB.x << " " << cB.y << "\n";
            else fc << "nan nan\n";

            // Force on B (total and normal); on A, the opposite
            const double FBx = fn * n.x + ft * t.x;
            const double FBy = fn * n.y + ft * t.y;
            const double FnBx = fn * n.x;
            const double FnBy = fn * n.y;
            const b2Body *bodies[2] = {bA, bB};
            const double sgn[2] = {-1.0, 1.0};
            for (int k = 0; k < 2; ++k) {
                if (!is_active_grain(bodies[k])) continue;
                auto it = grains.find(body_data(bodies[k])->gID);
                if (it == grains.end()) continue;
                GrainAcc &g = it->second;
                const b2Vec2 ctr = bodies[k]->GetWorldCenter();
                const double lx = static_cast<double>(p.x) - ctr.x;
                const double ly = static_cast<double>(p.y) - ctr.y;
                const double fx = sgn[k] * FBx, fy = sgn[k] * FBy;
                const double fnx = sgn[k] * FnBx, fny = sgn[k] * FnBy;
                g.s[0] += fx * lx;
                g.s[1] += fx * ly;
                g.s[2] += fy * lx;
                g.s[3] += fy * ly;
                g.sn[0] += fnx * lx;
                g.sn[1] += fnx * ly;
                g.sn[2] += fny * lx;
                g.sn[3] += fny * ly;
                g.F[0] += fx;
                g.F[1] += fy;
                g.tau += lx * fy - ly * fx;
            }
        }

        // Contact count (manifolds, not points)
        if (fn_sum > 0.0) {
            const b2Body *bodies[2] = {bA, bB};
            const bool other_is_wall[2] = {!gB, !gA};
            for (int k = 0; k < 2; ++k) {
                if (!is_active_grain(bodies[k])) continue;
                auto it = grains.find(body_data(bodies[k])->gID);
                if (it == grains.end()) continue;
                GrainAcc &g = it->second;
                ++g.z;
                if (other_is_wall[k]) ++g.z_w;
                if (n_active == 2) ++g.z_f;
            }
        }
    }
    fc.close();

    const std::string tensorName =
        jam_path(gs, "atasco_tensor_", jamIndex, ".dat");
    std::ofstream ft(tensorName.c_str());
    if (!ft.is_open()) {
        std::cout << "ERROR: could not create " << tensorName << std::endl;
        return;
    }
    ft << "# t: " << std::setprecision(10) << tSim << " dt: " << dt
       << " atasco: " << jamIndex << "\n"
       << "# s_ij = (1/A) sum_c F_i l_j, F: contact force on the grain "
          "(impulse/dt), l: contact point - center of mass. Compression < 0.\n"
       << "# sn: normal part of F only (the tangential part is s - sn). Includes "
          "contacts with walls. A: Box2D area (m / density).\n"
       << "# Fx Fy: net contact force; tau: net contact torque "
          "(l x F) = A (syx - sxy). At rest: F + m g = 0, tau = 0.\n"
       << "# z: contacts with Fn > 0 (manifolds, not points); z_w: of which with "
          "walls; z_f: of which with two active points (face-face).\n"
       << "# vx vy w: linear and angular velocity (to discard grains that are not "
          "at rest).\n"
       << "# gID tipo nLados x y A m sxx sxy syx syy snxx snxy snyx snyy "
          "Fx Fy tau z z_w z_f vx vy w\n";
    ft << std::scientific << std::setprecision(9);
    for (const auto &[gid, g] : grains) {
        const BodyData *bd = body_data(g.body);
        const b2Vec2 ctr = g.body->GetWorldCenter();
        ft << gid << " " << bd->tipo << " " << bd->nLados << " " << ctr.x
           << " " << ctr.y << " " << g.area << " " << g.body->GetMass();
        for (double v : g.s) ft << " " << v / g.area;
        for (double v : g.sn) ft << " " << v / g.area;
        const b2Vec2 v = g.body->GetLinearVelocity();
        ft << " " << g.F[0] << " " << g.F[1] << " " << g.tau << " " << g.z
           << " " << g.z_w << " " << g.z_f << " " << v.x << " " << v.y << " "
           << g.body->GetAngularVelocity() << "\n";
    }
    ft.close();
    std::cout << "# Clog stress tensor saved: " << tensorName
              << std::endl;
}
