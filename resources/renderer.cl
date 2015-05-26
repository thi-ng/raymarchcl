typedef struct {
  float3 pos;
  float3 dir;
} TRay;

typedef struct {
  float3 pos;
  float3 normal;
  float distance;
  int objectID;
  int iter;
} TIsec;

typedef struct {
  float4 albedo;
  float r0;
  float smoothness;
  float2 dummy;
} TMaterial;

typedef struct {
  float4 vertices[4];
  float4 normal;
  float4 color;
} TAreaLight;

typedef struct {
  float3 eyePos;
  float4 mcPos;
  float3 mcNormal;
  float2 pixelPos;
  float seed;
} TRenderState;

typedef struct {
  float3 eyePos;
  float3 targetPos;
  float3 up;
  float3 voxelBounds;
  float3 voxelBounds2;
  float3 voxelBoundsMin;
  float3 voxelBoundsMax;
  float3 invVoxelScale;
  float3 skyColor1;
  float3 skyColor2;
  int4 voxelRes;
  int2 resolution;
  float invAspect;
  float time;
  float fov;
  int maxIter;
  int maxVoxelIter;
  float maxDist;
  float startDist;
  float eps;
  int aoIter;
  float aoStepDist;
  float aoAmp;
  float voxelSize;
  float groundY;
  int shadowIter;
  int reflectIter;
  float shadowBias;
  float lightScatter;
  float minLightAtt;
  float gamma;
  float exposure;
  float dof;
  float frameBlend;
  float fogPow;
  float flareAmp;
  int mcTableLength;
  uchar isoVal;
  uchar numLights;
  float4 lightPos[4];
  float4 lightColor[4];
  TMaterial materials[4];
} TRenderOpts;

__constant float INV8BIT = 1.0f / 255.0f;
__constant int3 ONE = (int3)(1,0,-1);

float4 randFloat4(__global const float4* mcSamples, uint seed);
float2 distUnion(const float2 v1, const float2 v2);
float intersectsBox(const float3 bmin, const float3 bmax, const float3 p, const float3 dir);
uchar voxelLookup(__global const uchar* voxels, __private const TRenderOpts* opts, const float3 p);
float voxelLookupI(__global const uchar* voxels, __private const TRenderOpts* opts, const int3 q);
float3 voxelNormal(__global const uchar* voxels, __private const TRenderOpts* opts, const int3 q);
float3 voxelNormalSmooth(__global const uchar* voxels, __private const TRenderOpts* opts, const int3 q);
float voxelMaterial(__private const TRenderOpts* opts, const uchar v);
float2 distanceToScene(__global const uchar* voxels, __private const TRenderOpts* opts, TIsec* isec,
                       const float3 rpos, const float3 dir, int steps, bool smooth);
void raymarch(__global const uchar* voxels,
              __private const TRenderOpts* opts,
              const TRay* ray, TIsec* result, const float maxDist, int maxSteps, bool smooth);
float3 skyGradient(__private const TRenderOpts* opts, const float3 dir);
float3 lightPos(__global const float4* mcSamples,
                __private const TRenderOpts* opts,
                __private const TRenderState* state,
                const int i);
float3 reflect(const float3 v, const float3 n);
float3 applyAtmosphere(__global const float4* mcSamples,
                       __private const TRenderOpts* opts,
                       __private const TRenderState* state,
                       const TRay* ray, const TIsec* isec, float3 col);
float shadow(__global const uchar* voxels,
             __private const TRenderOpts* opts,
             const float3 p, const float3 ldir, const float ldist);
float schlick(const float r0, const float smooth, const float3 normal, const float3 view);
float diffuseIntensity(const float3 ldir, const float3 normal);
float blinnPhongIntensity(const float smooth, const TRay* ray,
                          const float3 lightDir, const float3 normal);
float ambientOcclusion(__global const uchar* voxels,
                       __global const float4* mcSamples,
                       __private const TRenderOpts* opts,
                       const float3 pos, const float3 normal);
float3 objectLighting(__global const uchar* voxels,
                      __global const float4* mcSamples,
                      __private const TRenderOpts* opts,
                      __private const TRenderState* state,
                      __private const TRay* ray,
                      __private TIsec* isec,
                      __private const TMaterial* mat,
                      const float3 normal, const float3 reflectCol);
float3 basicSceneColor(__global const uchar* voxels,
                       __global const float4* mcSamples,
                       const TRenderOpts* opts,
                       const TRenderState* state,
                       const TRay* ray, TIsec* isec);
float3 sceneColor(__global const uchar* voxels,
                  __global const float4* mcSamples,
                  const TRenderOpts* opts,
                  const TRenderState* state, const TRay* ray);
float3 gamma(const float3 col);
float3 tonemap(const float3 col, const float g);
TRay cameraRayLookat(const TRenderOpts* opts, const TRenderState* state);
TRenderState initRenderState(const TRenderOpts* opts,
                             __global const float4* mcSamples, const int id);

/* ---- implementation ---- */

float4 randFloat4(__global const float4* mcSamples, uint seed) {
  return mcSamples[seed & 0x3fff];
}

float2 distUnion(const float2 v1, const float2 v2) {
  return v1.x < v2.x ? v1 : v2;
}

/**
 * @returns distance to box or -1 if no isec
 */
float intersectsBox(const float3 bmin, const float3 bmax, const float3 p, const float3 dir) {
  const float3 omin = (bmin - p) / dir;
  const float3 omax = (bmax - p) / dir;
  float3 m = min(omax, omin);
  const float a = max(max(m.x, 0.0f), max(m.y, m.z));
  m = max(omax, omin);
  const float b = min(m.x, min(m.y, m.z));
  return b > a ? a : -1.0f;
}

uchar voxelLookup(__global const uchar* voxels, __private const TRenderOpts* opts, const float3 p) {
  const int4 res = opts->voxelRes;
  const int3 q = (int3)(p * res.xyz);
  if (q.z >= 0 && q.z < res.z && q.y >= 0 && q.y < res.y && q.x >= 0 && q.x < res.x) {
    return voxels[q.z * res.w + q.y * res.x + q.x];
  }
  return 0;
}

float voxelLookupI(__global const uchar* voxels, __private const TRenderOpts* opts, const int3 q) {
  const int4 res = opts->voxelRes;
  if (q.z >= 0 && q.z < res.z && q.y >= 0 && q.y < res.y && q.x >= 0 && q.x < res.x) {
    return step((float)opts->isoVal, (float)voxels[q.z * res.w + q.y * res.x + q.x]);
  }
  return 0.0f;
}

float3 voxelNormal(__global const uchar* voxels, __private const TRenderOpts* opts, const int3 q) {
  float nx = voxelLookupI(voxels, opts, q + ONE.xyy)
    - voxelLookupI(voxels, opts, q - ONE.xyy);
  float ny = voxelLookupI(voxels, opts, q + ONE.yxy)
    - voxelLookupI(voxels, opts, q - ONE.yxy);
  float nz = voxelLookupI(voxels, opts, q + ONE.yyx)
    - voxelLookupI(voxels, opts, q - ONE.yyx);
  return -(float3)(nx, ny, nz);
}

float3 voxelNormalSmooth(__global const uchar* voxels, __private const TRenderOpts* opts, const int3 q) {
  float3 n = (float3)0.0f;
  for(char z = -1; z <= 1; z++) {
    for(char y = -1; y <= 1; y++) {
      for(char x = -1; x <= 1; x++) {
        int3 qq = (int3)(q.x + x, q.y + y, q.z + z);
        if (voxelLookupI(voxels, opts, qq) > 0.0f) {
          n += voxelNormal(voxels, opts, qq);
        }
      }
    }
  }
  return normalize(n);
}

float voxelMaterial(__private const TRenderOpts* opts, const uchar v) {
  return (v < 168 ? (v < 84 ? 1.0f : 2.0f) : 3.0f);
}

float2 distanceToScene(__global const uchar* voxels, __private const TRenderOpts* opts, TIsec* isec,
                       const float3 rpos, const float3 dir, int steps, bool smooth) {
  float2 res = distUnion((float2)(rpos.y + opts->groundY), (float2)(1e5f, -1.0f));
  isec->normal = (res.x < 1e5) ? (float3)(0.0f, 1.0f, 0.0f) : -dir;
  const float idist = intersectsBox(opts->voxelBoundsMin, opts->voxelBoundsMax, rpos, dir);
  if (idist >= 0.0f && idist < res.x) {
    const float3 delta = dir / (steps * 0.5f) * opts->invVoxelScale;
    float3 p = rpos + opts->voxelBounds;
    if (idist > 0.0f) p = mad(dir, (float3)idist, p);
    p *= opts->invVoxelScale;
    while(--steps >= 0) {
      const uchar v = voxelLookup(voxels, opts, p);
      if (v > opts->isoVal) {
        const int4 vres = opts->voxelRes;
        const int3 q = (int3)(p * vres.xyz);
        if (smooth) {
          isec->normal = voxelNormalSmooth(voxels, opts, q);
        } else {
          isec->normal = normalize(voxelNormal(voxels, opts, q));
        }
        return distUnion((float2)(length(rpos - mad(p, opts->voxelBounds2, -opts->voxelBounds)) - opts->voxelSize,
                                  voxelMaterial(opts, v)), res);
      }
      p += delta;
    }
  }
  return res;
}

void raymarch(__global const uchar* voxels,
              __private const TRenderOpts* opts,
              const TRay* ray, TIsec* result, const float maxDist, int maxSteps, bool smooth) {
  result->distance = opts->startDist;
  while(--maxSteps >= 0) {
    result->pos = ray->pos + ray->dir * result->distance;
    const float2 sceneDist = distanceToScene(voxels, opts, result, result->pos, ray->dir, opts->maxVoxelIter, smooth);
    result->objectID = (int)sceneDist.y;
    if(fabs(sceneDist.x) <= opts->eps || result->distance >= maxDist) {
      break;
    }
    result->distance += sceneDist.x;
  }
  if(result->distance >= maxDist) {
    result->pos = ray->pos + ray->dir * result->distance;
    result->objectID = -1;
    result->distance = 1000.0f;
  }
}

float3 skyGradient(__private const TRenderOpts* opts, const float3 dir) {
  return mix(opts->skyColor1, opts->skyColor2, (float3)(dir.y * 0.5f + 0.5f));
}

float3 lightPos(__global const float4* mcSamples,
                __private const TRenderOpts* opts,
                __private const TRenderState* state,
                const int i) {
  uint seed = (uint)(state->pixelPos.x * 1957.0f + state->pixelPos.y * 2173.0f + opts->time * 4763.742f);
  return mad(randFloat4(mcSamples, seed).xyz, (float3)(opts->lightScatter), opts->lightPos[i].xyz);
}

float3 reflect(const float3 v, const float3 n){
  return v - 2.0f * dot(v, n) * n;
}

float3 applyAtmosphere(__global const float4* mcSamples,
                       __private const TRenderOpts* opts,
                       __private const TRenderState* state,
                       const TRay* ray,
                       const TIsec* isec,
                       float3 col) {
  const float3 fa = (float3)(1.0f - exp(isec->distance * isec->distance * -opts->fogPow));
  col = mad(skyGradient(opts, ray->dir) - col, fa, col);
  for(uchar i=0; i < opts->numLights; i++) {
    float3 lp = lightPos(mcSamples, opts, state, i);
    const float3 d = (float3)(clamp(dot(lp - ray->pos, ray->dir), 0.0f, isec->distance));
    lp = mad(ray->dir, d, ray->pos - lp);
    col = mad(opts->lightColor[i].xyz, (float3)(opts->flareAmp / dot(lp,lp)), col);
  }
  return col;
}

float shadow(__global const uchar* voxels,
             __private const TRenderOpts* opts,
             const float3 p, const float3 ldir, const float lightMaxDist) {
  TRay shadowRay;
  shadowRay.pos = p;
  shadowRay.dir = ldir;
  TIsec shadowIsec;
  raymarch(voxels, opts, &shadowRay, &shadowIsec, lightMaxDist, opts->shadowIter, false);
  return step(lightMaxDist, shadowIsec.distance);
}

// http://en.wikipedia.org/wiki/Schlick's_approximation
float schlick(const float r0, const float smooth, const float3 normal, const float3 view) {
  const float d = clamp(1.0f - dot(normal, -view), 0.0f, 1.0f);
  if (d > 0.0f) {
    const float d2 = d * d;
    return mad(1.0f - r0, smooth * d2 * d2 * d, r0);
  }
  return 0.0f;
}

float diffuseIntensity(const float3 ldir, const float3 normal) {
  return max(0.0f, dot(ldir, normal));
}

float blinnPhongIntensity(const float smooth, const TRay* ray,
                          const float3 lightDir, const float3 normal) {
  const float nh = dot(normalize(lightDir - ray->dir), normal);
  if (nh > 0.0f) {
    const float specPow = exp2(mad(6.0f, smooth, 4.0f));
    return pow(nh, specPow) * (specPow + 2.0f) * 0.125f;
  }
  return 0.0f;
}

float ambientOcclusion(__global const uchar* voxels,
                       __global const float4* mcSamples,
                       __private const TRenderOpts* opts,
                       const float3 pos,
                       const float3 normal) {
  float ao = 1.0f;
  float3 d = (float3)0.0f;
  uint seed = (uint)(pos.x * 3183.75f + pos.y * 1831.42f + pos.z * 2945.87f + opts->time * 2671.918f);
  const float3 scatter = (float3)0.2f;
  const float3 ad = (float3)opts->aoStepDist;
  TIsec isec;
  for(int i = 0; i <= opts->aoIter && ao > 0.01; i++) {
    d += ad;
    seed += 37;
    const float3 n = normalize(mad(randFloat4(mcSamples, seed).xyz, scatter, normal));
    const float2 sceneDist = distanceToScene(voxels, opts, &isec, mad(n, d, pos), n, opts->maxVoxelIter / 4, false);
    ao *= 1.0f - max((d.x - sceneDist.x) * opts->aoAmp / d.x, 0.0f);
  }
  return ao;
}

float3 objectLighting(__global const uchar* voxels,
                      __global const float4* mcSamples,
                      __private const TRenderOpts* opts,
                      __private const TRenderState* state,
                      __private const TRay* ray,
                      __private TIsec* isec,
                      __private const TMaterial* mat,
                      const float3 normal,
                      const float3 reflectCol) {
  float ao = ambientOcclusion(voxels, mcSamples, opts, isec->pos, normal);
  float3 diffReflect = skyGradient(opts, normal) * ao;
  float3 specReflect = reflectCol * ao;
  float3 finalCol = (float3)0.0f;
  for(uchar i = 0; i < opts->numLights; i++) {
    // point light
    float3 deltaLight = lightPos(mcSamples, opts, state, i) - isec->pos;
    float lightDist = dot(deltaLight, deltaLight);
    float att = 1.0f / lightDist;
    if (att > opts->minLightAtt) {
      float3 lightDir = normalize(deltaLight);
      float shadowFactor = shadow(voxels, opts, isec->pos + lightDir * opts->shadowBias,
                                  lightDir, min(sqrt(lightDist) - opts->shadowBias, opts->maxDist));
      if (shadowFactor > 0.0f) {
        float3 incidentLight = opts->lightColor[i].xyz * shadowFactor * att;
        diffReflect += diffuseIntensity(lightDir, normal) * incidentLight;
        specReflect += blinnPhongIntensity(mat->smoothness, ray, lightDir, normal) * incidentLight;
      }
    }
    diffReflect *= mat->albedo.xyz;
    // specular
    finalCol += mix(diffReflect, specReflect, schlick(mat->r0, mat->smoothness, normal, ray->dir));
  }
  return finalCol / (float)(opts->numLights);
}

float3 basicSceneColor(__global const uchar* voxels,
                       __global const float4* mcSamples,
                       const TRenderOpts* opts,
                       const TRenderState* state,
                       const TRay* ray,
                       TIsec* isec) {
  raymarch(voxels, opts, ray, isec, opts->maxDist, opts->maxIter, false);
  float3 sceneCol;
  if(isec->objectID < 0) {
    sceneCol = skyGradient(opts, ray->dir);
  } else {
    const TMaterial* mat = &opts->materials[isec->objectID];
    //TMaterial m2 = *mat;
    //isec->normal = sceneNormal(voxels, opts, isec->pos, ray->dir);
    //if (isec->objectID > 0) {
    //m2.albedo = (float4)(isec->pos + 1.0f, 0.0f);
    //m2.albedo = (float4)(mad(isec->normal.zyx, 2.0, 2.0f), 0.0f);
    //}
    sceneCol = objectLighting(voxels, mcSamples, opts, state, ray, isec, mat, isec->normal,
                              skyGradient(opts, reflect(ray->dir, isec->normal)));
  }
  return applyAtmosphere(mcSamples, opts, state, ray, isec, sceneCol);
}

float3 sceneColor(__global const uchar* voxels,
                  __global const float4* mcSamples,
                  const TRenderOpts* opts,
                  const TRenderState* state,
                  const TRay* ray) {
  TIsec isec;
  raymarch(voxels, opts, ray, &isec, opts->maxDist, opts->maxIter, true);
  float3 sceneCol;
  if(isec.distance >= opts->maxDist) {
    sceneCol = skyGradient(opts, ray->dir);
  } else {
    const TMaterial* mat = &opts->materials[isec.objectID];
    //TMaterial m2 = *mat;
    float3 norm = mad(state->mcNormal, 1.0f / mad(mat->smoothness, 200.0f, 5.0f), isec.normal);
    float3 reflectCol = (float3)mat->r0;
    //if (isec.objectID > 0) {
    //m2.albedo = (float4)(isec.pos + 1.0f, 0.0);
    //m2.albedo = (float4)(mad(norm.zyx, 2.0f, 2.0f), 0.0f);
    //}
    if (mat->r0 > 0.0f && opts->reflectIter > 0) {
      TIsec rIsec;
      rIsec.pos = isec.pos;
      rIsec.normal = norm;
      TRay reflectRay;
      reflectRay.dir = ray->dir;
      for(int i = 0; i < opts->reflectIter; i++) {
        reflectRay.dir = reflect(reflectRay.dir, rIsec.normal);
        reflectRay.pos = rIsec.pos + reflectRay.dir * 0.0075f;
        reflectCol += basicSceneColor(voxels, mcSamples, opts, state, &reflectRay, &rIsec);
        if (rIsec.objectID < 0) break;
        if ((&opts->materials[rIsec.objectID])->r0 < 0.001) break;
      }
    } else {
      reflectCol = skyGradient(opts, reflect(ray->dir, norm));
    }
    sceneCol = objectLighting(voxels, mcSamples, opts, state, ray, &isec, mat, norm, reflectCol);
  }
  sceneCol = applyAtmosphere(mcSamples, opts, state, ray, &isec, sceneCol);
  return sceneCol;
}

float3 gamma(const float3 col) {
  return col * col;
}

float3 tonemap(const float3 col, const float g) {
  return gamma(col / (g + col));
}

TRay cameraRayLookat(const TRenderOpts* opts, const TRenderState* state) {
  float3 forward = normalize(opts->targetPos - state->eyePos);
  float3 right = normalize(cross(forward, opts->up));
  float2 viewCoord = state->pixelPos / (float2)(opts->resolution.x, opts->resolution.y) * opts->fov - opts->fov * 0.5f;
  viewCoord.y *= -opts->invAspect;
  TRay ray;
  ray.pos = state->eyePos;
  ray.dir = normalize(right * viewCoord.x + cross(right, forward) * viewCoord.y + forward);
  return ray;
}

TRenderState initRenderState(const TRenderOpts* opts,
                             __global const float4* mcSamples, const int id) {
  float2 p = (float2)(id % opts->resolution.x, id / opts->resolution.x);
  TRenderState state;
  state.mcPos = randFloat4(mcSamples, (uint)(id * 17) + (uint)(opts->time * 3141.3862f));
  state.mcNormal = normalize(randFloat4(mcSamples, (uint)(id * 37) + (uint)(opts->time * 1859.1467f)).xyz);
  state.pixelPos = p + state.mcPos.zw;
  state.eyePos = mad(state.mcNormal.zxy, (float3)opts->dof, opts->eyePos);
  return state;
}

__kernel void RenderImage(__global const uchar* voxels,
                          __global const float4* mcSamples,
                          __global const TRenderOpts* g_opts,
                          __global float4* pixels,
                          const int n) {
  const int id = get_global_id(0);
  if (id < n) {
    __private TRenderOpts opts;
    __private TRenderState state;
    __private TRay ray;
    opts = *g_opts;
    state = initRenderState(&opts, mcSamples, id);
    ray = cameraRayLookat(&opts, &state);
    const float3 sceneCol = sceneColor(voxels, mcSamples, &opts, &state, &ray) * opts.exposure;
    pixels[id] = (float4)(mix(pixels[id].xyz, sceneCol, opts.frameBlend), 1.0f);
  }
}

__kernel void TonemapImage(__global const float4* pixels,
                           __global TRenderOpts* opts,
                           __global uint* rgba,
                           const int n) {
  int id = get_global_id(0);
  if (id < n) {
    float3 col = tonemap(pixels[id].xyz, opts->gamma) * 255.0f;
    rgba[id] = 0xff000000 |
      ((int)(clamp(col.x, 0.0f, 255.0f)) << 16) |
      ((int)(clamp(col.y, 0.0f, 255.0f)) << 8) |
      (int)(clamp(col.z, 0.0f, 255.0f));
  }
}
