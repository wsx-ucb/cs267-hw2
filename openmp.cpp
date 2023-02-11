#include "common.h"
#include <omp.h>
#include <cmath>
#include <vector>
using std::vector;

static int bn;
static vector<vector<particle_t*>> bins1;
static vector<vector<particle_t*>> bins2;
static vector<omp_lock_t> locks;

void classify(particle_t* p, vector<vector<particle_t*>>& bins, double size) {
    int bi = p->x / cutoff;
    int bj = p->y / cutoff;
    int idx = bi * bn + bj;
    // omp_set_lock(&locks[idx]);
    bins[idx].push_back(p);
    // omp_unset_lock(&locks[idx]);
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    bn = ceil(size / cutoff);
    bins1.resize(bn * bn);
    bins2.resize(bn * bn);
    locks.resize(bn * bn);
    for (int i = 0; i < bn * bn; i++) {
        omp_init_lock(&locks[i]);
    }
    for (int i = 0; i < num_parts; i++) {
        classify(parts + i, bins1, size);
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

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    #pragma omp barrier
    #pragma omp for collapse(2) nowait
    for (int i = 0; i < bn; i++) {
        for (int j = 0; j < bn; j++) {
            for (particle_t* p : bins1[i * bn + j]) {
                p->ax = p->ay = 0;
                for (int ni : {i-1, i, i+1}) {
                    if (ni < 0 || ni >= bn) continue;
                    for (int nj : {j-1, j, j+1}) {
                        if (nj < 0 || nj >= bn || (ni == i && nj == j)) continue;
                        for (particle_t* n : bins1[ni * bn + nj]) {
                            apply_force(*p, *n);
                        }
                    }
                }
                for (particle_t* n : bins1[i * bn + j]) {
                    if (n == p) continue;
                    apply_force(*p, *n);
                }
            }
        }
    }
    #pragma omp for
    for (int i = 0; i < bn * bn; i++) {
        bins2[i].clear();
    }
    #pragma omp for nowait
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }
    #pragma omp master
    {
        for (int i = 0; i < num_parts; i++) {
            classify(parts + i, bins2, size);
        }
        std::swap(bins1, bins2);
    }
}
