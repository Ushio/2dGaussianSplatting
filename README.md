## 2d gaussian splatting for 2d image reconstruction  

![demo](demo.apng)

## build

```
git submodule update --init
premake5 vs2022
```

## references
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- (Ja) https://www.youtube.com/watch?v=VyVi7iPb1Uw
- (En) https://www.youtube.com/watch?v=e50Bj7jn9IQ


## Formulation
It doesn't use any of auto-diff tools, here is the analytical form of several derivatives.

So my formulation is [here](Form.pdf)
