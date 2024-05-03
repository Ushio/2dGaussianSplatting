#include "pr.hpp"
#include <iostream>
#include <memory>

#include <intrin.h>

uint32_t pcg(uint32_t v)
{
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

glm::uvec3 pcg3d(glm::uvec3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

glm::vec3 sign_of(glm::vec3 v)
{
    return {
        v.x < 0.0f ? -1.0f : 1.0f,
        v.y < 0.0f ? -1.0f : 1.0f,
        v.z < 0.0f ? -1.0f : 1.0f
    };
}
float sign_of(float v)
{
    return v < 0.0f ? -1.0f : 1.0f;
}

struct Splat
{
    glm::vec2 pos;
	float sx;
	float sy;
	float rot;
    glm::vec3 color;
};

#define POS_PURB 0.1f
#define RADIUS_PURB 0.1f
#define COLOR_PURB 0.01f

#define RADIUS_MAX 16.0f

enum
{
    SIGNBIT_POS_X = 0,
    SIGNBIT_POS_Y,
    SIGNBIT_RADIUS,
    SIGNBIT_COL_R,
    SIGNBIT_COL_G,
    SIGNBIT_COL_B,
};

bool bitAt(uint32_t u, uint32_t i)
{
    return u & (1u << i);
}

// 0: +1, 1: -1
float signAt(uint32_t u, uint32_t i)
{
    return bitAt(u, i) ? -1.0f : 1.0f;
}

uint32_t splatRng(uint32_t i, uint32_t perturbIdx)
{
    return pcg(i + pcg(perturbIdx));
}

float lengthSquared(glm::vec2 v)
{
    return v.x * v.x + v.y * v.y;
}
float lengthSquared(glm::vec3 v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}
//void drawSplats(pr::Image2DRGBA32* image, std::vector<int>* splatIndices, const std::vector<Splat>& splats, uint32_t perturbIdx, float s)
//{
//    int w = image->width();
//    int h = image->height();
//    for (int i = 0; i < splats.size(); i++)
//    {
//        Splat splat = splats[i];
//
//        // Apply perturb
//        splat = perturb(splat, splatRng(i, perturbIdx), s);
//
//        glm::ivec2 lower = glm::ivec2(glm::floor(splat.pos - glm::vec2(splat.radius, splat.radius)));
//        glm::ivec2 upper = glm::ivec2(glm::ceil(splat.pos + glm::vec2(splat.radius, splat.radius)));
//
//        lower = glm::clamp(lower, glm::ivec2(0, 0), glm::ivec2(w - 1, h - 1));
//        upper = glm::clamp(upper, glm::ivec2(0, 0), glm::ivec2(w - 1, h - 1));
//
//        for (int y = lower.y; y <= upper.y; y++)
//        {
//            for (int x = lower.x; x <= upper.x; x++)
//            {
//                float d2 = lengthSquared(splat.pos - glm::vec2((float)x, (float)y));
//                if (d2 < splat.radius * splat.radius)
//                {
//                    float T = std::expf(-2 * d2 / (splat.radius * splat.radius));
//                    glm::vec3 c = (*image)(x, y);
//                    c = glm::mix(c, splat.color, T);
//                    (*image)(x, y) = glm::vec4(c, 1.0f);
//
//                    if (splatIndices)
//                    {
//                        splatIndices[y * w + x].push_back(i);
//                    }
//                }
//            }
//        }
//    }
//}


const float ADAM_BETA1 = 0.9f;
const float ADAM_BETA2 = 0.99f;

struct Adam
{
    float m_m;
    float m_v;

    float optimize(float value, float g, float alpha, float beta1t, float beta2t)
    {
        float s = alpha;
        float m = ADAM_BETA1 * m_m + (1.0f - ADAM_BETA1) * g;
        float v = ADAM_BETA2 * m_v + (1.0f - ADAM_BETA2) * g * g;
        m_m = m;
        m_v = v;
        float m_hat = m / (1.0f - beta1t);
        float v_hat = v / (1.0f - beta2t);

        const float ADAM_E = 1.0e-15f;
        return value - s * m_hat / (sqrt(v_hat) + ADAM_E);
    }
};
struct SplatAdam
{
    Adam pos[2];
	Adam sx;
	Adam sy;
	Adam rot;
    Adam color[3];
};

template <class T>
inline T ss_max( T x, T y )
{
	return ( x < y ) ? y : x;
}

template <class T>
inline T ss_min( T x, T y )
{
	return ( y < x ) ? y : x;
}

// ax^2 + bx + c == 0
int solve_quadratic( float xs[2], float a, float b, float c )
{
	float det = b * b - 4.0f * a * c;
	if( det < 0.0f )
	{
		return 0;
	}

	float k = ( -b - sign_of( b ) * std::sqrtf( det ) ) / 2.0f;
	float x0 = k / a;
	float x1 = c / k;
	xs[0] = ss_min( x0, x1 );
	xs[1] = ss_max( x0, x1 );
	return 2;
}

void eignValues( float* lambda0, float* lambda1, float* determinant, const glm::mat2& mat )
{
	float mean = ( mat[0][0] + mat[1][1] ) * 0.5f;
	float det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
	float d = std::sqrtf( ss_max( mean * mean - det, 0.0f ) );
	*lambda0 = mean + d;
	*lambda1 = mean - d;
    *determinant = det;
}

glm::mat2 rot2d( float rad )
{
	float cosTheta = std::cosf( rad );
	float sinTheta = std::sinf( rad );
	return glm::mat2( cosTheta, sinTheta, -sinTheta, cosTheta );
};

// \sigma = V * L * V^(-1)
glm::mat2 cov_of( const Splat& splat )
{
	float theta = splat.rot;
	float sx = splat.sx;
	float sy = splat.sy;

	// for consistent order
	if( splat.sx < splat.sy )
	{
		theta += glm::pi<float>();
		std::swap( sx, sy );
	}

	float cosTheta = std::cosf( theta );
	float sinTheta = std::sinf( theta );
	float lambda0 = sx * sx;
	float lambda1 = sy * sy;
	float s11 = lambda0 * cosTheta * cosTheta + lambda1 * sinTheta * sinTheta;
	float s12 = ( lambda0 - lambda1 ) * sinTheta * cosTheta;
	return glm::mat2(
		s11, s12,
		s12, lambda0 + lambda1 - s11 );
}

void eigen_vectors_of_cov( glm::vec2* eigen0, glm::vec2* eigen1, const glm::mat2& cov, float lambda0 /*larger*/ )
{
	float s11 = cov[0][0];
	float s22 = cov[1][1];
	float s12 = cov[1][0];

	float eps = 1e-15f;
	glm::vec2 e0 = glm::normalize( s11 < s22 ? glm::vec2( s12 + eps, lambda0 - s11 ) : glm::vec2( lambda0 - s22, s12 + eps ) );
	glm::vec2 e1 = { -e0.y, e0.x };
	*eigen0 = e0;
	*eigen1 = e1;
}

int main() {
    using namespace pr;

    SetDataDir(ExecutableDir());

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 0, 0, 200 };
    camera.lookat = { 0, 0, 0 };

    double e = GetElapsedTime();

    ITexture* textureRef = CreateTexture();
    Image2DRGBA32 imageRef;
    {
        Image2DRGBA8 image;
        image.load("squirrel_cls_mini.jpg");
        imageRef = Image2DRGBA8_to_Image2DRGBA32(image);
    }
    // std::fill(imageRef.data(), imageRef.data() + imageRef.width() * imageRef.height(), glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
    //for (int y = 0; y < imageRef.height(); y++)
    //{
    //    for (int x = 0; x < imageRef.width(); x++)
    //    {
    //        imageRef(x, y) = glm::vec4((float)x / imageRef.width(), 1- (float)x / imageRef.width(), 0.0f, 1.0f);
    //    }
    //}

    textureRef->upload(imageRef);

    int NSplat = 512;
	std::vector<Splat> splats( NSplat );
	
    for( int i = 0; i < splats.size(); i++ )
	{
		glm::vec3 r0 = glm::vec3( pcg3d( { i, 0, 0xFFFFFFFF } ) ) / glm::vec3( 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );
		glm::vec3 r1 = glm::vec3( pcg3d( { i, 1, 0xFFFFFFFF } ) ) / glm::vec3( 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );

		Splat s;
		s.pos.x = glm::mix( r0.x, (float)imageRef.width() - 1, r0.x );
		s.pos.y = glm::mix( r0.y, (float)imageRef.height() - 1, r0.y );
		//s.sx = glm::mix( 4.0f, 8.0f, r1.x );
		//s.sy = glm::mix( 4.0f, 8.0f, r1.y );
		s.sx = 4;
		s.sy = 8;
		s.rot = glm::pi<float>() * 2.0f * r1.z;
		s.color = { 0.5f, 0.5f, 0.5f };
		splats[i] = s;
	}

    float beta1t = 1.0f;
    float beta2t = 1.0f;
    std::vector<SplatAdam> splatAdams(splats.size());

	beta1t = 1.0f;
	beta2t = 1.0f;
	splatAdams.clear();
	splatAdams.resize( NSplat );


    ITexture* tex0 = CreateTexture();
	Image2DRGBA32 image0;
	image0.allocate( imageRef.width(), imageRef.height() );

	Image2DRGBA32 image1;
	image1.allocate( imageRef.width(), imageRef.height() );

    //std::vector<std::vector<int>> indices0(imageRef.width() * imageRef.height());
    //std::vector<int> indices1(imageRef.width() * imageRef.height());

    // drawSplats(&image0, splats, 0 );

    //for (int y = 0; y < image0.height(); y++)
    //{
    //    for (int x = 0; x < image0.width(); x++)
    //    {
    //        glm::vec3 r0 = glm::vec3(pcg3d({ x, y, 0 })) / glm::vec3(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    //        image0(x, y) = glm::vec4(r0, 1.0f);
    //    }
    //}

    // tex0->upload(image0);

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XY, 10.0f, 20, { 128, 128, 128 });
        DrawXYZAxis(10.0f);

        DrawCube({ image0.width() * 0.5f, -image0.height() * 0.5f, 0 }, { image0.width(),  image0.height(), 0 }, { 255,255,255 });
        
        // static glm::vec3 begP = { 50, 50, 0 };
		// static glm::vec3 endP = { 100, 80, 0 };

        static glm::vec3 splat_p = { 50, 50, 0 };
		static float splat_sx  = 8;
		static float splat_sy  = 10;
		static float splat_rot = 0.0f;

        auto man2d = [camera](glm::vec3* p, float manipulatorSize ) {
            p->y = -p->y;
            ManipulatePosition(camera, p, manipulatorSize);
            p->z = 0.0f;
            p->y = -p->y;
        };
		man2d( &splat_p, 10 );
		// man2d(&endP, 10);

        //std::fill( image0.data(), image0.data() + image0.width() * image0.height(), glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );
		// drawLineDDA( &image0, begP, endP, glm::vec4( 1.0f, 1.0f, 1.0f, 1.0f ) );

  //      auto rot2d = []( float rad ) {
		//	float cosTheta = std::cosf( rad );
		//	float sinTheta = std::sinf( rad );
		//	return glm::mat2( cosTheta, sinTheta, -sinTheta, cosTheta);
		//};

  //      glm::mat2 R = rot2d( splat_rot );
  //      glm::mat2 cov = R * glm::mat2(
		//	splat_sx * splat_sx, 0.0f,
		//	0.0f, splat_sy * splat_sy
  //      ) * glm::transpose(R);

  //      float det;
		//float lamda0;
		//float lamda1;
		//eignValues( &lamda0, &lamda1, &det, cov );

  //      glm::mat2 inv_cov =
		//	glm::mat2(
		//		cov[1][1], -cov[0][1],
		//		-cov[1][0], cov[0][0] ) / det; 

  //      glm::vec2 eigen0, eigen1; 
  //      eigen_vectors_of_cov( &eigen0, &eigen1, cov, lamda0 );

  //      // printf( "%f\n", s12 + lamda0 );
		//// printf( "%f %f\n", eigen0.x, eigen0.y );

  //      float sqrt_of_lamda0 = std::sqrtf( lamda0 );
		//float sqrt_of_lamda1 = std::sqrtf( lamda1 );

		//for( int y = 0; y < image0.height(); y++ )
		//{
		//	for( int x = 0; x < image0.width(); x++ )
		//	{
		//		glm::vec2 p = { x + 0.5f, y + 0.5f };
		//		glm::vec2 v = p - glm::vec2( splat_p );
		//		float g = std::expf( -0.5f * glm::dot( v, inv_cov * v ) );
		//		image0( x, y ) = glm::vec4( g, g, g, 1.0f );
		//	}
		//}
		//drawLineDDA( &image0, glm::vec2( splat_p ), glm::vec2( splat_p ) + eigen0 * sqrt_of_lamda0, glm::vec4( 1.0f, 0.0f, 0.0f, 1.0f ) );
		//drawLineDDA( &image0, glm::vec2( splat_p ), glm::vec2( splat_p ) + eigen1 * sqrt_of_lamda1, glm::vec4( 1.0f, 0.0f, 0.0f, 1.0f ) );

        std::fill( image0.data(), image0.data() + image0.width() * image0.height(), glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );

        // forward
        for( int i = 0; i < splats.size(); i++ )
		{
			Splat s = splats[i];

            glm::mat2 cov = cov_of( s );

			float det;
			float lambda0;
			float lambda1;
			eignValues( &lambda0, &lambda1, &det, cov );
			float sqrt_of_lambda0 = std::sqrtf( lambda0 );
			float sqrt_of_lambda1 = std::sqrtf( lambda1 );

			glm::mat2 inv_cov =
				glm::mat2(
					cov[1][1], -cov[0][1],
					-cov[1][0], cov[0][0] ) /
				det; 

            glm::vec2 eigen0, eigen1;
			eigen_vectors_of_cov( &eigen0, &eigen1, cov, lambda0 );

            glm::vec2 axis0 = eigen0 * sqrt_of_lambda0;
			glm::vec2 axis1 = eigen1 * sqrt_of_lambda1;
			DrawEllipse( { s.pos.x, -s.pos.y, 0 }, { axis0.x, -axis0.y, 0.0f }, { axis1.x, -axis1.y, 0.0f }, { 255, 255, 255 } );

            float r = ss_max( sqrt_of_lambda0, sqrt_of_lambda1 ) * 3.0f;
			int begX = s.pos.x - r;
			int endX = s.pos.x + r;
			int begY = s.pos.y - r;
			int endY = s.pos.y + r;
			for( int y = begY; y <= endY; y++ )
			{
				if( y < 0 || image0.height() <= y )
					continue;

				for( int x = begX; x <= endX; x++ )
				{
					if( x < 0 || image0.width() <= x )
						continue;

                    // w as throughput
                    glm::vec4 color = image0( x, y );
					float T = color.w;

					glm::vec2 p = { x + 0.5f, y + 0.5f };
					glm::vec2 v = p - s.pos;
					float alpha = std::expf( -0.5f * glm::dot( v, inv_cov * v ) );

					color.x += T * s.color.x * alpha;
					color.y += T * s.color.y * alpha;
					color.z += T * s.color.z * alpha;

                    color.w *= ( 1.0f - alpha );

					image0( x, y ) = color;
				}
			}
		}

		// clear throughput
		for (int i = 0; i < image0.width() * image0.height(); i++)
		{
			image0.data()[i].w = 1.0f;
		}

        // backward
		std::fill( image1.data(), image1.data() + image1.width() * image1.height(), glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );
		std::vector<Splat> dSplats( splats.size() );

		for( int i = 0; i < splats.size(); i++ )
		{
			Splat s = splats[i];

			glm::mat2 cov = cov_of( s );

			float det;
			float lambda0;
			float lambda1;
			eignValues( &lambda0, &lambda1, &det, cov );
			float sqrt_of_lambda0 = std::sqrtf( lambda0 );
			float sqrt_of_lambda1 = std::sqrtf( lambda1 );

			glm::mat2 inv_cov =
				glm::mat2(
					cov[1][1], -cov[0][1],
					-cov[1][0], cov[0][0] ) /
				det;

			glm::vec2 eigen0, eigen1;
			eigen_vectors_of_cov( &eigen0, &eigen1, cov, lambda0 );

			glm::vec2 axis0 = eigen0 * sqrt_of_lambda0;
			glm::vec2 axis1 = eigen1 * sqrt_of_lambda1;
			DrawEllipse( { s.pos.x, -s.pos.y, 0 }, { axis0.x, -axis0.y, 0.0f }, { axis1.x, -axis1.y, 0.0f }, { 255, 255, 255 } );

			float r = ss_max( sqrt_of_lambda0, sqrt_of_lambda1 ) * 3.0f;
			int begX = s.pos.x - r;
			int endX = s.pos.x + r;
			int begY = s.pos.y - r;
			int endY = s.pos.y + r;
			for( int y = begY; y <= endY; y++ )
			{
				if( y < 0 || image0.height() <= y )
					continue;

				for( int x = begX; x <= endX; x++ )
				{
					if( x < 0 || image0.width() <= x )
						continue;

					// w as throughput
					glm::vec4 color = image1( x, y );
					float T = color.w;

					glm::vec2 p = { x + 0.5f, y + 0.5f };
					glm::vec2 v = p - s.pos;
					float alpha = std::expf( -0.5f * glm::dot( v, inv_cov * v ) );

					glm::vec4 finalColor = image0( x, y );

					// dL/dc
					glm::vec3 dL_dC = glm::vec3( finalColor - imageRef( x, y ) );
					{
						float dC_dc = alpha * T /* throughput */;
						dSplats[i].color += dL_dC * dC_dc;
					}

					// color accumuration
					color.x += T * s.color.x * alpha;
					color.y += T * s.color.y * alpha;
					color.z += T * s.color.z * alpha;

					glm::vec3 S = finalColor - color;
					// printf( "%.5f %.5f %.5f\n", S.x / ( 1.0f - alpha ), S.y / ( 1.0f - alpha ), S.z / ( 1.0f - alpha ) );
					{
						glm::vec3 dC_dalpha = s.color * T - S / ( 1.0f - alpha );
						float a = inv_cov[0][0];
						float b = inv_cov[1][0];
						float c = inv_cov[0][1];
						float d = inv_cov[1][1];
						float dalpha_dx = 0.5f * alpha * ( 2.0f * a * v.x + ( b + c ) * v.y );
						float dalpha_dy = 0.5f * alpha * ( 2.0f * d * v.y + ( b + c ) * v.x );

						// numerical varidation x this is just for v not mu
						//float eps = 0.00001f;
						//float da =
						//	( std::expf( -0.5f * glm::dot( v + glm::vec2( eps, 0.0f ), inv_cov * (v + glm::vec2( eps, 0.0f )) ) ) - std::expf( -0.5f * glm::dot( v, inv_cov * v ) ) ) / eps;
						//printf( "%.5f %.5f\n", dalpha_dx, da );
						
						// numerical varidation y
						//float eps = 0.00001f;
						//float da =
						//	( std::expf( -0.5f * glm::dot( v + glm::vec2( 0.0f, eps ), inv_cov * ( v + glm::vec2( 0.0f, eps ) ) ) ) - std::expf( -0.5f * glm::dot( v, inv_cov * v ) ) ) / eps;
						//printf( "%.5f %.5f\n", dalpha_dy, da );

						dSplats[i].pos.x +=
							dalpha_dx * dC_dalpha.x * dL_dC.x +
							dalpha_dx * dC_dalpha.y * dL_dC.y +
							dalpha_dx * dC_dalpha.z * dL_dC.z;
						dSplats[i].pos.y +=
							dalpha_dy * dC_dalpha.x * dL_dC.x +
							dalpha_dy * dC_dalpha.y * dL_dC.y +
							dalpha_dy * dC_dalpha.z * dL_dC.z;

						float lambda0sq = lambda0 * lambda0;
						float lambda1sq = lambda1 * lambda1;
						float cosTheta = std::cosf( s.rot );
						float sinTheta = std::sinf( s.rot );
						float da_dlambda0 = -1.0f / lambda0sq * cosTheta * cosTheta;
						float da_dlambda1 = -1.0f / lambda1sq * sinTheta * sinTheta;

						float dbc_dlambda0 = -1.0f / lambda0sq * sinTheta * cosTheta;
						float dbc_dlambda1 = +1.0f / lambda1sq * sinTheta * cosTheta;

						float dd_dlambda0 = -1.0f / lambda0sq * sinTheta * sinTheta;
						float dd_dlambda1 = -1.0f / lambda1sq * cosTheta * cosTheta;

						float dlambda0_dsx = 2.0f * s.sx;
						float dlambda1_dsy = 2.0f * s.sy;

						float dalpha_dsx = -0.5f * alpha * ( v.x * v.x * da_dlambda0 + v.x * v.y * dbc_dlambda0 * 2.0f + v.y * v.y * dd_dlambda0 ) * dlambda0_dsx;
						float dalpha_dsy = -0.5f * alpha * ( v.x * v.x * da_dlambda1 + v.x * v.y * dbc_dlambda1 * 2.0f + v.y * v.y * dd_dlambda1 ) * dlambda1_dsy;

						// numerical varidation
						//float eps = 0.00001f; 
						//Splat ds = s;
						//ds.sx += eps;
						//float derivative = ( std::expf( -0.5f * glm::dot( v, glm::inverse( cov_of( ds ) ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_dsx, derivative );

						//float eps = 0.00001f;
						//Splat ds = s;
						//ds.sy += eps;
						//float derivative = ( std::expf( -0.5f * glm::dot( v, glm::inverse( cov_of( ds ) ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_dsy, derivative );

						dSplats[i].sx +=
							dL_dC.x * dC_dalpha.x * dalpha_dsx +
							dL_dC.y * dC_dalpha.y * dalpha_dsx +
							dL_dC.z * dC_dalpha.z * dalpha_dsx;
						dSplats[i].sy +=
							dL_dC.x * dC_dalpha.x * dalpha_dsy +
							dL_dC.y * dC_dalpha.y * dalpha_dsy +
							dL_dC.z * dC_dalpha.z * dalpha_dsy;

						float da_dtheta = 2.0f * ( lambda0 - lambda1 ) / ( lambda0 * lambda1 ) * sinTheta * cosTheta;
						float db_dtheta = -( lambda0 - lambda1 ) / ( lambda0 * lambda1 ) * ( cosTheta * cosTheta - sinTheta * sinTheta );
						float dd_dtheta = -da_dtheta;

						float dalpha_dtheta =
							-0.5f * alpha * (
								da_dtheta * v.x * v.x +
								2.0f * db_dtheta * v.x * v.y + 
								dd_dtheta * v.y * v.y
							);

						dSplats[i].rot +=
							dL_dC.x * dC_dalpha.x * dalpha_dtheta +
							dL_dC.y * dC_dalpha.y * dalpha_dtheta +
							dL_dC.z * dC_dalpha.z * dalpha_dtheta;

						// numerical varidation
						//float eps = 0.001f;
						//Splat ds = s;
						//ds.rot += eps;
						//float derivative = ( std::expf( -0.5f * glm::dot( v, glm::inverse( cov_of( ds ) ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_dtheta, derivative );

						//float derivative = ( glm::inverse( cov_of( ds ) )[0][0] - a ) / eps;
						//printf( "%f %f\n", da_dtheta, derivative );

						//float derivative = ( glm::inverse( cov_of( ds ) )[1][0] - b ) / eps;
						//printf( "%f %f\n", db_dtheta, derivative );

						//float derivative = ( glm::inverse( cov_of( ds ) )[1][1] - d ) / eps;
						//printf( "%f %f\n", dd_dtheta, derivative );
					}

					color.w *= ( 1.0f - alpha );

					image1( x, y ) = color;
				}
			}
		}

		// optimize
		float trainingRate = 0.01f;

		// gradient decent
		beta1t *= ADAM_BETA1;
		beta2t *= ADAM_BETA2;

		for( int i = 0; i < splats.size(); i++ )
		{
			splats[i].color.x = splatAdams[i].color[0].optimize( splats[i].color.x, dSplats[i].color.x, trainingRate, beta1t, beta2t );
			splats[i].color.y = splatAdams[i].color[1].optimize( splats[i].color.y, dSplats[i].color.y, trainingRate, beta1t, beta2t );
			splats[i].color.z = splatAdams[i].color[2].optimize( splats[i].color.z, dSplats[i].color.z, trainingRate, beta1t, beta2t );

			splats[i].pos.x = splatAdams[i].pos[0].optimize( splats[i].pos.x, dSplats[i].pos.x, trainingRate, beta1t, beta2t );
			splats[i].pos.y = splatAdams[i].pos[1].optimize( splats[i].pos.y, dSplats[i].pos.y, trainingRate, beta1t, beta2t );

			splats[i].sx = splatAdams[i].sx.optimize( splats[i].sx, dSplats[i].sx, trainingRate, beta1t, beta2t );
			splats[i].sy = splatAdams[i].sy.optimize( splats[i].sy, dSplats[i].sy, trainingRate, beta1t, beta2t );

			splats[i].rot = splatAdams[i].rot.optimize( splats[i].rot, dSplats[i].rot, trainingRate, beta1t, beta2t );

			// constraints
			splats[i].pos.x = glm::clamp( splats[i].pos.x, 0.0f, (float)imageRef.width() - 1 );
			splats[i].pos.y = glm::clamp( splats[i].pos.y, 0.0f, (float)imageRef.height() - 1 );
			splats[i].sx = ss_max( splats[i].sx, 1.0f );
			splats[i].sy = ss_max( splats[i].sy, 1.0f );

			splats[i].color = glm::clamp( splats[i].color, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } );
		}


        // clear throughput
		for( int i = 0; i < image0.width() * image0.height(); i++ )
		{
			image0.data()[i].w = 1.0f;
			image1.data()[i].w = 1.0f;
		}

        tex0->upload(image0);

		double mse = 0.0;
		for( int y = 0; y < image0.height(); y++ )
		{
			for( int x = 0; x < image0.width(); x++ )
			{
				glm::vec3 d = image0( x, y ) - imageRef( x, y );
				mse += lengthSquared( d * 255.0f );
			}
		}
		mse /= ( image0.height() * image0.width() * 3 );


        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowPos({ 20, 20 }, ImGuiCond_Once);
        ImGui::SetNextWindowSize({ 600, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
		ImGui::Text( "mse = %.5f", mse );
		ImGui::SliderFloat( "sx", &splat_sx, 0, 64 );
		ImGui::SliderFloat( "sy", &splat_sy, 0, 64 );
		ImGui::SliderFloat( "rot", &splat_rot, -glm::pi<float>(), glm::pi<float>() );
   
        ImGui::Image(textureRef, ImVec2(textureRef->width() * 2, textureRef->height() * 2));
        ImGui::Image(tex0, ImVec2(tex0->width() * 2, tex0->height() * 2));

        ImGui::End();

        //ImGui::SetNextWindowPos({ 800, 20 }, ImGuiCond_Once);
        //ImGui::SetNextWindowSize({ 600, 300 }, ImGuiCond_Once);
        //ImGui::Begin("Params");
        //ImGui::SliderFloat("scale", &scale, 0, 1);
        //ImGui::End();

        EndImGui();
    }

    pr::CleanUp();

    return 0;
}
