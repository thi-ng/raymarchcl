typedef struct {
  float3 pos;
  float3 dir;
} TRay;

typedef struct {
  float3 pos;
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
  float3 lightPos;
  float3 lightColor;
  int3 voxelRes;
  int2 resolution;
  float invAspect;
  float time;
  int maxIter;
  int maxVoxelIter;
  float maxDist;
  float startDist;
  float eps;
  int aoIter;
  float aoStepDist;
  float aoAmp;
  float voxelSize;
  float isoVal;
  float groundY;
  int shadowIter;
  float shadowBias;
  float lightScatter;
  float minLightAtt;
  float gamma;
  float exposure;
  float dof;
  float frameBlend;
  float fogDensity;
  float flareAmp;
  float4 normOffsets[4];
  TMaterial materials[4];
  float4 mcSamples[16384];
} TRenderOptions;

float4 randFloat4(global const TRenderOptions* opts, uint seed) {
  return opts->mcSamples[seed & 0x3fff];
}

float4 distUnion(const float4 v1, const float4 v2) {
  return v1.x < v2.x ? v1 : v2;
}

float trilinear(float n000, float n001, float n010, float n011,
                float n100, float n101, float n110, float n111,
                float fx, float fy, float fz) {
  float nx = mix(n000, n100, fx);
  float ny = mix(n001, n101, fx);
  float nz = mix(n010, n110, fx);
  float nw = mix(n011, n111, fx);
  return mix(mix(nx, nz, fy), mix(ny, nw, fy), fz);
}

/**
 * @returns distance to box or -1 if no isec
 */
float intersectsBox(const float3 bmin, const float3 bmax, const float3 p, const float3 dir) {
  float3 omin = (bmin - p) / dir;
  float3 omax = (bmax - p) / dir;
  float3 m = min(omax, omin);
  float a = max(max(m.x, 0.0f), max(m.y, m.z));
  m = max(omax, omin);
  float b = min(m.x, min(m.y, m.z));
  return b > a ? a : -1.0f;
}

float voxelDataAt(global const uchar* voxels, const int3 res, const int3 p) {
  if (p.x >= 0 && p.x < res.x && p.y >= 0 && p.y < res.y && p.z >= 0 && p.z < res.z) {
    return (float)(voxels[p.z * res.x * res.y + p.y * res.x + p.x]) / 255.0;
  }
  return 0.0f;
}

float voxelLookup(global const uchar* voxels, global const TRenderOptions* opts, const float3 p) {
  int3 pv = (int3)(p.x * opts->voxelRes.x, p.y * opts->voxelRes.y, p.z * opts->voxelRes.z);
  return voxelDataAt(voxels, opts->voxelRes, pv);
}
/*
float voxelDataAt(global const uchar* voxels, const int3 res, const int3 p) {
  return voxels[p.z * res.x * res.y + p.y * res.x + p.x];
}

float voxelLookup(global const uchar* voxels, global const TRenderOptions* opts, const float3 p) {
  int3 res = opts->voxelRes;
  float3 pv = (float3)(p.x * res.x, p.y * res.y, p.z * res.z);
  int3 pi = (int3)(pv.x, pv.y, pv.z);
  if (pi.x >= 0 && pi.x < res.x-1 && pi.y >= 0 && pi.y < res.y-1 && pi.z >= 0 && pi.z < res.z-1) {
    int3 qi = pi + (int3)(1,1,1);
    float n000 = voxelDataAt(voxels, res, pi);
    float n001 = voxelDataAt(voxels, res, (int3)(pi.x, pi.y, qi.z));
    float n010 = voxelDataAt(voxels, res, (int3)(pi.x, qi.y, pi.z));
    float n011 = voxelDataAt(voxels, res, (int3)(pi.x, pi.y, qi.z));
    float n100 = voxelDataAt(voxels, res, (int3)(qi.x, pi.y, pi.z));
    float n101 = voxelDataAt(voxels, res, (int3)(qi.x, pi.y, qi.z));
    float n110 = voxelDataAt(voxels, res, (int3)(qi.x, qi.y, pi.z));
    float n111 = voxelDataAt(voxels, res, (int3)(qi.x, qi.y, qi.z));
    return trilinear(n000, n001, n010, n011, n100, n101, n110, n111,
                     p.x-(float)(pi.x), p.y-(float)(pi.y), p.z-(float)(pi.z));
  }
  return 0.0f;
}
*/
float voxelMaterial(global const TRenderOptions* opts, float v) {
  return (v < 0.66 ? (v < 0.33 ? 1.0f : 2.0f) : 3.0f);
}

float4 distanceToScene(global const uchar* voxels, global const TRenderOptions* opts,
                       const float3 rpos, const float3 dir, const int steps) {
  float4 res = distUnion((float4)(rpos.y + opts->groundY, 0.0, rpos.xz), (float4)(10000.0f, -1.0f, 0.0f, 0.0f));
  float idist = intersectsBox(opts->voxelBoundsMin, opts->voxelBoundsMax, rpos, dir);
  if (idist >= 0.0f && idist < res.x) {
    float3 delta = dir / (float3)(steps/2.0f) * opts->invVoxelScale;
    float3 p = rpos + opts->voxelBounds;
    if (idist > 0.0f) p += dir * idist;
    p *= opts->invVoxelScale;
    for(int i = 0; i < steps; i++) {
      float v = voxelLookup(voxels, opts, p);
      if (v > opts->isoVal) {
        float d = length(rpos - (p * opts->voxelBounds2 - opts->voxelBounds)) - opts->voxelSize;
        return distUnion((float4)(d, voxelMaterial(opts, v), 0.0f, 0.0f), res);
      }
      p += delta;
    }
  }
  return res;
}

float3 sceneNormal(global const uchar* voxels, global const TRenderOptions* opts,
                   const float3 p, const float3 dir) {
  const float f1 = distanceToScene(voxels, opts, p + opts->normOffsets[0].xyz, dir, opts->maxVoxelIter).x;
  const float f2 = distanceToScene(voxels, opts, p + opts->normOffsets[1].xyz, dir, opts->maxVoxelIter).x;
  const float f3 = distanceToScene(voxels, opts, p + opts->normOffsets[2].xyz, dir, opts->maxVoxelIter).x;
  const float f4 = distanceToScene(voxels, opts, p + opts->normOffsets[3].xyz, dir, opts->maxVoxelIter).x;
  return normalize(opts->normOffsets[0].xyz * f1 + opts->normOffsets[1].xyz * f2 +
                   opts->normOffsets[2].xyz * f3 + opts->normOffsets[3].xyz * f4);
}

void raymarch(global const uchar* voxels, global const TRenderOptions* opts,
              const TRay* ray, TIsec* result, const float maxDist, const int maxSteps) {
  result->distance = opts->startDist;
  result->objectID = 0;
  result->iter = 1;
  for(int i = 0; i < maxSteps; i++) {
    result->pos = ray->pos + ray->dir * result->distance;
    float4 sceneDist = distanceToScene(voxels, opts, result->pos, ray->dir, opts->maxVoxelIter);
    result->objectID = (int)(sceneDist.y);
    if(fabs(sceneDist.x) <= opts->eps || result->distance >= maxDist) {
      break;
    }
    result->distance += sceneDist.x;
    result->iter++;
  }
  if(result->distance >= maxDist) {
    result->pos = ray->pos + ray->dir * result->distance;
    result->objectID = -1;
    result->distance = 1000.0f;
  }
}

TMaterial objectMaterial(global const TRenderOptions* opts, const int objectID) {
  return opts->materials[objectID];
}

float3 skyGradient(global const TRenderOptions* opts, const float3 dir) {
  return mix(opts->skyColor1, opts->skyColor2, dir.y * 0.5f + 0.5f);
}

float3 lightPos(global const TRenderOptions* opts, const TRenderState* state) {
  uint seed = (uint)(state->pixelPos.x * 1957.0f + state->pixelPos.y * 2173.0f + opts->time * 4763.742f);
  return opts->lightPos + randFloat4(opts, seed).xyz * opts->lightScatter;
}

float3 reflect(const float3 v, const float3 n){
  return v - 2.0f * dot(v, n) * n;
}

float3 applyAtmosphere(global const TRenderOptions* opts,
                       const TRenderState* state, const TRay* ray, const TIsec* isec, float3 col) {
  float fa = exp(isec->distance * isec->distance * -opts->fogDensity);
  float3 fogCol = skyGradient(opts, ray->dir);
  col = (float3)(mix(fogCol.x, col.x, fa),
                 mix(fogCol.y, col.y, fa),
                 mix(fogCol.z, col.z, fa));
  float3 lp = lightPos(opts, state);
  float d = clamp(dot(lp - ray->pos, ray->dir), 0.0f, isec->distance);
  lp = ray->pos + ray->dir * d - lp;
  col += opts->lightColor * opts->flareAmp / dot(lp,lp);
  return col;
}

float shadow(global const uchar* voxels, global const TRenderOptions* opts,
             const float3 p, const float3 ldir, const float ldist) {
  TRay shadowRay;
  shadowRay.pos = p;
  shadowRay.dir = ldir;
  TIsec shadowIsec;
  raymarch(voxels, opts, &shadowRay, &shadowIsec, ldist, opts->shadowIter);
  return shadowIsec.distance > 0.0f ? step(ldist, shadowIsec.distance) : 0.0f;
}

// http://en.wikipedia.org/wiki/Schlick's_approximation
float schlick(const TMaterial mat, const float3 normal, const float3 view) {
  float d = clamp(1.0f - dot(normal, -view), 0.0f, 1.0f);
  float d2 = d * d;
  return mat.r0 + (1.0f - mat.r0) * mat.smoothness * d2 * d2 * d;
}

float diffuseIntensity(const float3 ldir, const float3 normal) {
  return max(0.0f, dot(ldir, normal));
}

float blinnPhongIntensity(const TMaterial mat, const TRay* ray,
                          const float3 lightDir, const float3 normal) {
  float3 vHalf = normalize(lightDir - ray->dir);
  float nh = dot(vHalf, normal);
  if (nh > 0.0f) {
    float specPow = exp2(4.0f + 6.0f * mat.smoothness);
    return pow(nh, specPow) * (specPow + 2.0f) * 0.125f;
  }
  return 0.0f;
}

float ambientOcclusion(global const uchar* voxels, global const TRenderOptions* opts,
                       const float3 pos, const float3 normal) {
  float ao = 1.0f;
  float d = 0.0f;
  uint seed = (uint)(pos.x * 3183.75f + pos.y * 1831.42f + pos.z * 2945.87f + opts->time * 2671.918f);
  for(int i = 0; i <= opts->aoIter && ao > 0.01; i++) {
    d += opts->aoStepDist;
    float3 n = normalize(normal + 0.2f * randFloat4(opts, seed + i * 37).xyz);
    float4 sceneDist = distanceToScene(voxels, opts, pos + n * d, n, opts->maxIter);
    ao *= 1.0f - max(0.0f, (d - sceneDist.x) * opts->aoAmp / d );
  }
  return ao;
}

float3 objectLighting(global const uchar* voxels, global const TRenderOptions* opts,
                      const TRenderState* state, const TRay* ray, TIsec* isec,
                      const TMaterial mat, const float3 normal, const float3 reflectCol) {
  float ao = ambientOcclusion(voxels, opts, isec->pos, normal);
  float3 diffReflect = skyGradient(opts, normal) * ao;
  float3 specReflect = reflectCol * ao;
  // point light
  float3 deltaLight = lightPos(opts, state) - isec->pos;
  float lightDist = dot(deltaLight, deltaLight);
  float att = 1.0f / lightDist;
  if (att > opts->minLightAtt) {
    float3 lightDir = normalize(deltaLight);
    float shadowFactor = shadow(voxels, opts, isec->pos + lightDir * opts->shadowBias,
                                lightDir, min(sqrt(lightDist) - opts->shadowBias, opts->maxDist));
    float3 incidentLight = opts->lightColor * shadowFactor * att;
    diffReflect += diffuseIntensity(lightDir, normal) * incidentLight;
    specReflect += blinnPhongIntensity(mat, ray, lightDir, normal) * incidentLight;
  }
  diffReflect *= mat.albedo.xyz;
  // specular
  float spec = schlick(mat, normal, ray->dir);
  float3 col = (float3)(mix(diffReflect.x, specReflect.x, spec),
                        mix(diffReflect.y, specReflect.y, spec),
                        mix(diffReflect.z, specReflect.z, spec));
  return col;
}

float3 basicSceneColor(global const uchar* voxels, global const TRenderOptions* opts,
                       const TRenderState* state, const TRay* ray) {
  TIsec isec;
  raymarch(voxels, opts, ray, &isec, opts->maxDist, opts->maxIter);

  float3 sceneCol;

  if(isec.objectID < 0) {
    sceneCol = skyGradient(opts, ray->dir);
  } else {
    const TMaterial mat = objectMaterial(opts, isec.objectID);
    float3 norm = sceneNormal(voxels, opts, isec.pos, ray->dir);
    // use sky gradient instead of reflection
    float3 reflectCol = skyGradient(opts, reflect(ray->dir, norm));
    // apply lighting
    sceneCol = objectLighting(voxels, opts, state, ray, &isec, mat, norm, reflectCol);
  }
  return applyAtmosphere(opts, state, ray, &isec, sceneCol);
}

float3 sceneColor(global const uchar* voxels, global const TRenderOptions* opts,
                  const TRenderState* state, const TRay* ray) {
  TIsec isec;
  raymarch(voxels, opts, ray, &isec, opts->maxDist, opts->maxIter);
  float3 sceneCol;
  if(isec.objectID < 0) {
    sceneCol = skyGradient(opts, ray->dir);
  } else {
    const TMaterial mat = objectMaterial(opts, isec.objectID);
    float3 norm = sceneNormal(voxels, opts, isec.pos, ray->dir);
    norm = normalize(norm + state->mcNormal / (5.0f + mat.smoothness * 200.0f));
    float3 reflectCol;
    if (mat.r0 > 0.0) {
      TRay reflectRay;
      reflectRay.dir = reflect(ray->dir, norm);
      reflectRay.pos = isec.pos + reflectRay.dir * 0.05f; // TODO opts->reflectSeperation
      reflectCol = basicSceneColor(voxels, opts, state, &reflectRay);
    } else {
      reflectCol = skyGradient(opts, reflect(ray->dir, norm));
    }
    sceneCol = objectLighting(voxels, opts, state, ray, &isec, mat, norm, reflectCol);
  }
  sceneCol = applyAtmosphere(opts, state, ray, &isec, sceneCol);
  return sceneCol;
}

float3 gamma(const float3 col) {
  return col * col;
}

float3 tonemap(const float3 col, const float g) {
  return gamma(col / (g + col));
}

TRay cameraRayLookat(global const TRenderOptions* opts, const TRenderState* state) {
  float3 forward = normalize(opts->targetPos - state->eyePos);
  float3 right = normalize(cross(forward, opts->up));
  float2 viewCoord = state->pixelPos / (float2)(opts->resolution.x, opts->resolution.y) * 2.0f - 1.0f;
  viewCoord.y *= -opts->invAspect;
  TRay ray;
  ray.pos = state->eyePos;
  ray.dir = normalize(right * viewCoord.x + cross(right, forward) * viewCoord.y + forward);
  return ray;
}

TRenderState initRenderState(global const TRenderOptions* opts, const int id) {
  float2 p = (float2)(id % opts->resolution.x, id / opts->resolution.x);
  TRenderState state;
  state.mcPos = randFloat4(opts, (uint)(id * 17) + (uint)(opts->time * 3141.3862f));
  state.mcNormal = normalize(randFloat4(opts, (uint)(id * 37) + (uint)(opts->time * 1859.1467f)).xyz);
  state.pixelPos = p + state.mcPos.zw;
  state.eyePos = opts->eyePos + state.mcNormal.zxy * opts->dof;
  return state;
}

__kernel void RenderImage(global const uchar* voxels,
                          global const TRenderOptions* opts,
                          global float4* pixels,
                          const int n) {
  int id = get_global_id(0);
  if (id < n) {
    TRenderState state = initRenderState(opts, id);
    TRay ray = cameraRayLookat(opts, &state);
    float3 sceneCol = sceneColor(voxels, opts, &state, &ray) * opts->exposure;
    float4 prevCol = pixels[id] + (state.mcPos - 0.4f) * (1.0f / 255.0f);
    float4 finalCol = (float4)(mix(prevCol.x, sceneCol.x, opts->frameBlend),
                               mix(prevCol.y, sceneCol.y, opts->frameBlend),
                               mix(prevCol.z, sceneCol.z, opts->frameBlend),
                               1.0f);
    pixels[id] = finalCol;
  }
}

__kernel void TonemapImage(global const float4* pixels,
                           global const TRenderOptions* opts,
                           global uint* rgba,
                           const int n) {
  int id = get_global_id(0);
  if (id < n) {
    float3 col = tonemap(pixels[id].xyz, opts->gamma);
    int r = (int)(min(max(col.x * 255.0f, 0.0f), 255.0f));
    int g = (int)(min(max(col.y * 255.0f, 0.0f), 255.0f));
    int b = (int)(min(max(col.z * 255.0f, 0.0f), 255.0f));
    rgba[id] = 0xff000000 | (r << 16) | (g << 8) | b;
  }
}
