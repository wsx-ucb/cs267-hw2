#include "common.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
using std::vector;

static int bn;
static vector<vector<particle_t*>> bins;
static vector<omp_lock_t> locks;

void bin(particle_t* p, int i, int j, bool locking) {
    int idx = i * bn + j;
    if (locking) omp_set_lock(&locks[idx]);
    bins[idx].push_back(p);
    if (locking) omp_unset_lock(&locks[idx]);
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    bn = ceil(size / cutoff);
    bins.resize(bn * bn);
    locks.resize(bn * bn);
    for (int i = 0; i < bn * bn; i++) {
        omp_init_lock(&locks[i]);
    }
    for (int i = 0; i < num_parts; i++) {
        particle_t* p = parts + i;
        int bi = p->x / cutoff;
        int bj = p->y / cutoff;
        bin(p, bi, bj, false);
    }
}

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void apply_force_2D(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
    neighbor.ax -= coef * dx;
    neighbor.ay -= coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

struct group {
    int imin, imax, jmin, jmax;
};

group get_group(int id, int total) {
    group g;
    int s = std::max((bn + total - 1) / total, 1);
    g.imin = std::min(id * s, bn);
    g.imax = std::min(g.imin + s, bn);
    g.jmin = 0;
    g.jmax = bn;
    return g;
}

bool in_group(int i, int j, group const& g) {
    return g.imin <= i && i < g.imax && g.jmin <= j && j < g.jmax;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    #pragma omp barrier
    group g = get_group(omp_get_thread_num(), omp_get_num_threads());
    // iterate through row order cells in group
    for (int i = g.imin; i < g.imax; i++) {
        for (int j = g.jmin; j < g.jmax; j++) {
            // iterate trhoguh particles a bin at i,j
            for (particle_t* p : bins[i * bn + j]) {
                // clear acceleration
                p->ax = p->ay = 0;
            }
        }
    }
    for (int i = g.imin; i < g.imax; i++) {
        for (int j = g.jmin; j < g.jmax; j++) {
            // iterate trhoguh particles a bin at i,j
            vector<particle_t*> bin = bins[i * bn + j]; //this is useful when comparing particles in the same bin
            for (int p_i=0; p_i<bin.size; p_i++) {
                particle_t* p = bin[p_i];
                // comparision: iterate through rowsi-1 to i+1
                for (int ni : {i-1, i, i+1}) {
                    // skip comparison if not reasonable
                    if (ni < 0 || ni >= bn) continue;
                    // iterate through cols j-1 to j+1
                    for (int nj : {j-1, j, j+1}) {
                        // skip comparison if not reasonable
                        if (nj < 0 || nj >= bn) continue;
                        // iterate though bins to compare particles.
                        if ( (ni * bn + nj) < (i * bn + j) )    {
                            if (ni<g.imin || nj<g.jmin || nj>g.jmax)  {
                                for (particle_t* n : bins[ni * bn + nj]) {
                                    apply_force(*p, *n); //1-D
                                }
                            }
                        }                        

                        else if ( (ni * bn + nj) > (i * bn + j) )    {
                            if (ni>g.imax || nj>g.jmax || nj<g.jmin)  {
                                for (particle_t* n : bins[ni * bn + nj]) {
                                    apply_force(*p, *n); //1-D
                                }
                            }
                            else {
                                for (particle_t* n : bins[ni * bn + nj]) {
                                    apply_force_2D(*p, *n); //2-D
                                }
                            }
                        }

                        else {
                            for (int n_i=p_i+1; n_i<bin.size; n_i++) {
                                particle_t* n = bin[n_i];
                                    apply_force_2D(*p, *n); //2-D
                            }
                        }
                    }
                }
            }
        }
    }


    struct movement {
        particle_t* p;
        int i, j;
    };
    vector<movement> in_group_moves;
    vector<movement> out_group_moves;
    #pragma omp barrier
    for (int i = g.imin; i < g.imax; i++) {
        for (int j = g.jmin; j < g.jmax; j++) {
            vector<particle_t*>& b = bins[i * bn + j];
            b.erase(std::remove_if(b.begin(), b.end(), [i, j, size, &in_group_moves, &out_group_moves, &g](particle_t* p) {
                move(*p, size);
                int bi = p->x / cutoff;
                int bj = p->y / cutoff;
                if (bi == i && bj == j) return false;
                if (in_group(bi, bj, g)) {
                    in_group_moves.push_back({p, bi, bj});
                } else {
                    out_group_moves.push_back({p, bi, bj});
                }
                return true;
            }), b.end());
        }
    }
    for (movement m : in_group_moves) {
        bin(m.p, m.i, m.j, false);
    }
    #pragma omp barrier
    for (movement m : out_group_moves) {
        bin(m.p, m.i, m.j, true);
    }
}
