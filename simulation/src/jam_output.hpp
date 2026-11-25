/*! \file jam_output.hpp
 * \brief Output of each clog: contacts and stress tensor of each grain.
 *
 * These functions are called together with saveAtascoFrame, before the arch
 * is removed. At that moment the positions and the manifold impulses
 * correspond to the same (last) Step, so the three outputs are consistent.
 *
 * Box2D 2.4 convention (b2ContactSolver): the manifold normal n points from
 * body A to body B and the tangent is t = b2Cross(n, 1) = (n.y, -n.x). The
 * mean force on B during the step, at point i, is
 *
 *     F_B = (normalImpulse n + tangentImpulse t) / dt,    F_A = -F_B.
 *
 * Stress tensor of each grain p (Love formula, compression < 0):
 *
 *     s_ij = (1/A_p) sum_c F_i l_j,   l = contact point - center of mass,
 *
 * with A_p the area of the grain in Box2D (the one that gives its mass: for
 * polygons, the area of the vertices, without the b2_polygonRadius skin).
 * sn is the part due to the normal forces only; the tangential part is
 * s - sn (for polygons sn is not symmetric and friction contributes to the
 * trace).
 */

#ifndef _JAM_OUTPUT_H
#define _JAM_OUTPUT_H

#include <box2d/box2d.h>
#include "globalSetup.hpp"

/*! Writes frames_<dirID>/atasco_contacts_<jam>.dat: one row per manifold
 * point of each touching contact (grains and walls), "ID_A ID_B PosX PosY
 * NormalImpulse TangentImpulse", with impulses (not forces) and the normal
 * from A to B. This is the input of
 * analysis/3_presion_y_mapas/presion_tensor_arch.py. */
void saveAtascoContacts(b2World *world, const GlobalSetup *gs, int jamIndex);

/*! Computes the stress tensor of each grain in the simulation and writes:
 *  - frames_<dirID>/atasco_tensor_<jam>.dat: one row per active grain
 *    (isGrain and isIn, the same as in atasco_<jam>.xy);
 *  - frames_<dirID>/atasco_contacts_full_<jam>.dat: one row per manifold
 *    point, with the Box2D normal, the forces and the centers of mass
 *    (used by the analysis instead of inferring the normal from geometry).
 */
void saveAtascoTensor(b2World *world, const GlobalSetup *gs, int jamIndex,
                      double tSim);

#endif
