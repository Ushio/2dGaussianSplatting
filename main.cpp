#include "pr.hpp"
#include <iostream>
#include <memory>

#include <intrin.h>

#define SPLAT_BOUNDS 3.0f
#define MIN_THROUGHPUT ( 1.0f / 256.0f )

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
float exp_approx( float x )
{
	// return expf( glm::clamp( x, -16.0f, 16.0f ) ); // use this for numerical varidation

	/*
	float L = 0.0f;
	float R = 1.0f;
	for (int i = 0 ; i < 1000 ; i++)
	{
		float m = ( L + R ) * 0.5f;
		float x = m;
		x *= x;
		x *= x;
		x *= x;
		if( x == 0.0f || fpclassify(x) == FP_SUBNORMAL )
		{
			L = m;
		}
		else
		{
			R = m;
		}
	}
	printf( "%.32f\n", R ); >> 0.00001814586175896693021059036255
	*/
	x = 1.0f + x / 8.0f;
	if( x < 0.00001814586175896693021059036255f ) // avoid subnormal
	{
		return 0.0f;
	}
	x *= x;
	x *= x;
	x *= x;
	return x;
}

struct Splat
{
    glm::vec2 pos;
	glm::vec2 u;
	glm::vec2 v;
	//float sx;
	//float sy;
	//float rot;
    glm::vec3 color;
	float opacity;
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
	Adam u[2];
	Adam v[2];
	//Adam sx;
	//Adam sy;
	//Adam rot;
    Adam color[3];
	Adam opacity;
};



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

// lambda0 is larger
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
//glm::mat2 cov_of( const Splat& splat )
//{
//	float theta = splat.rot;
//	float sx = splat.sx;
//	float sy = splat.sy;
//
//	float cosTheta = std::cosf( theta );
//	float sinTheta = std::sinf( theta );
//	float lambda0 = sx * sx;
//	float lambda1 = sy * sy;
//	float s11 = lambda0 * cosTheta * cosTheta + lambda1 * sinTheta * sinTheta;
//	float s12 = ( lambda0 - lambda1 ) * sinTheta * cosTheta;
//	return glm::mat2(
//		s11, s12,
//		s12, lambda0 + lambda1 - s11 );
//}


//glm::mat2 inv_cov_of( const Splat& splat, float E )
//{
//	const glm::vec2& u = splat.u;
//	const glm::vec2& v = splat.v;
//	float a = u.x * u.x + v.x * v.x + E;
//	float b = u.x * u.y + v.x * v.y;
//	float d = u.y * u.y + v.y * v.y + E;
//	return glm::mat2(
//		a, b,
//		b, d );
//}
glm::mat2 cov_of( const Splat& splat )
{
	const float E = 1.0e-4f;
	const glm::vec2& u = splat.u;
	const glm::vec2& v = splat.v;
	float a = u.x * u.x + v.x * v.x + E;
	float b = u.x * u.y + v.x * v.y;
	float d = u.y * u.y + v.y * v.y + E;
	return glm::mat2(
		a, b,
		b, d );
}

void eigenVectors_of_symmetric( glm::vec2* eigen0, glm::vec2* eigen1, const glm::mat2& m, float lambda )
{
	float s11 = m[0][0];
	float s22 = m[1][1];
	float s12 = m[1][0];

	// to workaround lambda0 == lambda1
	float eps = 1e-15f;
	glm::vec2 e0 = glm::normalize( s11 < s22 ? glm::vec2( s12 + eps, lambda - s11 ) : glm::vec2( lambda - s22, s12 + eps ) );
	glm::vec2 e1 = { -e0.y, e0.x };
	*eigen0 = e0;
	*eigen1 = e1;
}

float sqr(float x)
{
	return x * x;
}


int main() {
    using namespace pr;

    SetDataDir(ExecutableDir());

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 0;
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

    int NSplat = 1024;
	std::vector<Splat> splats;

    float beta1t = 1.0f;
    float beta2t = 1.0f;
    std::vector<SplatAdam> splatAdams;

	int iterations = 0;

	auto init = [&]() {
		iterations = 0;

		beta1t = 1.0f;
		beta2t = 1.0f;
		splats.clear();
		splats.resize( NSplat );
		splatAdams.clear();
		splatAdams.resize( NSplat );

		for( int i = 0; i < splats.size(); i++ )
		{
			glm::vec3 r0 = glm::vec3( pcg3d( { i, 0, 0xFFFFFFFF } ) ) / glm::vec3( 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );
			glm::vec3 r1 = glm::vec3( pcg3d( { i, 1, 0xFFFFFFFF } ) ) / glm::vec3( 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );

			Splat s;
			s.pos.x = glm::mix( r0.x, (float)imageRef.width() - 1, r0.x );
			s.pos.y = glm::mix( r0.y, (float)imageRef.height() - 1, r0.y );
			// s.sx = glm::mix( 6.0f, 10.0f, r1.x );
			// s.sy = glm::mix( 6.0f, 10.0f, r1.y );
			// s.sx = 8;
			// s.sy = 8;
			// s.rot = glm::pi<float>() * r1.z;
			
			//if (trainableE)
			//{
			//	s.u = glm::vec2( 1.0f, 0.0f );
			//	s.v = glm::vec2( 0.0f, 1.0f );
			//}
			//else
			//{
			//	s.u = glm::vec2( 1.0f, 0.0f ) / glm::mix( 6.0f, 10.0f, r1.x );
			//	s.v = glm::vec2( 0.0f, 1.0f ) / glm::mix( 6.0f, 10.0f, r1.y );
			//}
			s.u = glm::vec2( 1, 0 );
			s.v = glm::vec2( 0, 1 );
			s.color = { 0.5f, 0.5f, 0.5f };
			s.opacity = 1.0f;
			splats[i] = s;
		}
	};


	init();

    ITexture* tex0 = CreateTexture();
	Image2DRGBA32 image0;
	image0.allocate( imageRef.width(), imageRef.height() );

	Image2DRGBA32 image1;
	image1.allocate( imageRef.width(), imageRef.height() );

	bool showSplatInfo = false;
	bool optimizeOpacity = false;
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

  //      static glm::vec3 splat_p = { 50, 50, 0 };
		//static float splat_sx  = 8;
		//static float splat_sy  = 10;
		//static float splat_rot = 0.0f;

  //      auto man2d = [camera](glm::vec3* p, float manipulatorSize ) {
  //          p->y = -p->y;
  //          ManipulatePosition(camera, p, manipulatorSize);
  //          p->z = 0.0f;
  //          p->y = -p->y;
  //      };
		//man2d( &splat_p, 10 );
		// 

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

		PrimBegin( PrimitiveMode::Lines, 1 );

        // forward
        for( int i = 0; i < splats.size(); i++ )
		{
			Splat s = splats[i];

   //         glm::mat2 cov = cov_of( s );

			//float det;
			//float lambda0;
			//float lambda1;
			//eignValues( &lambda0, &lambda1, &det, cov );
			//float sqrt_of_lambda0 = std::sqrtf( lambda0 );
			//float sqrt_of_lambda1 = std::sqrtf( lambda1 );

			glm::mat2 cov = cov_of( s );
			glm::mat2 inv_cov = glm::inverse( cov );

			float det_of_cov;
			float lambda0;
			float lambda1;
			eignValues( &lambda0, &lambda1, &det_of_cov, cov );

            glm::vec2 eigen0, eigen1;
			eigenVectors_of_symmetric( &eigen0, &eigen1, cov, lambda0 );

			// visuallize
			{
				glm::vec2 axis0 = eigen0 * std::sqrtf( lambda0 );
				glm::vec2 axis1 = eigen1 * std::sqrtf( lambda1 );

				//Draw axis
				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ), { 255, 255, 255 } );
				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ) + glm::vec3( axis0.x, -axis0.y, 0 ), { 255, 255, 255 } );

				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ), { 255, 255, 255 } );
				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ) + glm::vec3( axis1.x, -axis1.y, 0 ), { 230, 230, 230 } );
				
				// UV
				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ), { 255, 128, 128 } );
				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ) + glm::vec3( s.u.x, -s.u.y, 0 ), { 255, 128, 128 } );
				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ), { 128, 255, 128 } );
				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ) + glm::vec3( s.v.x, -s.v.y, 0 ), { 128, 255, 128 } );

				//Draw Ellipse
				int nvtx = 16;
				CircleGenerator circular( glm::pi<float>() * 2.0f / nvtx );
				glm::uvec3 col = s.color * 255.0f;
				for( int i = 0; i <= nvtx; i++ )
				{
					PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ) + glm::vec3( axis0.x, -axis0.y, 0.0f ) * circular.sin() + glm::vec3( axis1.x, -axis1.y, 0.0f ) * circular.cos(), col );
					circular.step();
					PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ) + glm::vec3( axis0.x, -axis0.y, 0.0f ) * circular.sin() + glm::vec3( axis1.x, -axis1.y, 0.0f ) * circular.cos(), col );
				}

				// Draw the exact bounding box from covariance matrix
				float hsize_invCovX = std::sqrt( inv_cov[1][1] * det_of_cov );
				float hsize_invCovY = std::sqrt( inv_cov[0][0] * det_of_cov );
				glm::vec3 vs[4] = {
					{ -hsize_invCovX, -hsize_invCovY, 0.0f },
					{ +hsize_invCovX, -hsize_invCovY, 0.0f },
					{ +hsize_invCovX, +hsize_invCovY, 0.0f },
					{ -hsize_invCovX, +hsize_invCovY, 0.0f },
				};
				for (int i = 0; i < 4; i++)
				{
					PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ) + vs[i], { 128, 128, 128 } );
					PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ) + vs[( i + 1 ) % 4], { 128, 128, 128 } );
				}

				if (showSplatInfo)
				{
					char op[128];
					sprintf( op, "o=%.2f, c=(%.2f, %.2f, %.2f)", s.opacity, s.color.x, s.color.y, s.color.z );
					DrawText( glm::vec3( s.pos.x, -s.pos.y, 0 ), op, 12 );
				}
			}

			// The exact bounding box from covariance matrix
			// float hsize_invCovX = std::sqrt( inv_cov[1][1] / det_of_inv ) * SPLAT_BOUNDS;
			float hsize_invCovY = std::sqrt( inv_cov[0][0] * det_of_cov ) * SPLAT_BOUNDS;
			int begY = ss_max( s.pos.y - hsize_invCovY, 0.0f );
			int endY = ss_min( s.pos.y + hsize_invCovY, image0.height() -1.0f );
			for( int y = begY; y <= endY; y++ )
			{
				if( y < 0 || image0.height() <= y )
					continue;

				// Minimum range of x
				float vy = ( y + 0.5f ) - s.pos.y;
				float a = inv_cov[0][0];
				float b = inv_cov[1][0];
				float d = inv_cov[1][1];
				float xs[2];
				int begX = -1;
				int endX = -1;
				if( solve_quadratic( xs, a, 2.0f * b * vy, d * vy * vy - SPLAT_BOUNDS * SPLAT_BOUNDS ) )
				{
					begX = s.pos.x + xs[0];
					endX = s.pos.x + xs[1];
				}

				for( int x = begX; x <= endX; x++ )
				{
					if( x < 0 || image0.width() <= x )
						continue;

                    // w as throughput
                    glm::vec4 color = image0( x, y );
					float T = color.w;

					if( T < MIN_THROUGHPUT )
						continue;

					glm::vec2 p = { x + 0.5f, y + 0.5f };
					glm::vec2 v = p - s.pos;

					float d2 = glm::dot( v, inv_cov * v );
					float alpha = exp_approx( -0.5f * d2 ) * s.opacity;

					color.x += T * s.color.x * alpha;
					color.y += T * s.color.y * alpha;
					color.z += T * s.color.z * alpha;

                    color.w *= ( 1.0f - alpha );

					image0( x, y ) = color;
				}
			}
		}

		PrimEnd();

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

			// glm::mat2 cov = cov_of( s );
			// glm::mat2 inv_cov = inv_cov_of( s, E );
			glm::mat2 cov = cov_of( s );
			glm::mat2 inv_cov = glm::inverse( cov );

			// det = det(cov) = 1 / det(inv_cov)
			// sx * sy is an alternative. but leave it as the 3d spatting case can't use original scale in screen space.
			//float det = cov[0][0] * cov[1][1] - cov[1][0] * cov[0][1];
			//glm::mat2 inv_cov =
			//	glm::mat2(
			//		cov[1][1], -cov[0][1],
			//		-cov[1][0], cov[0][0] ) /
			//	det;

			//float theta = s.rot;
			//float cosTheta = std::cosf( theta );
			//float sinTheta = std::sinf( theta );

			// The exact bounding box from covariance matrix
			float det_of_cov = glm::determinant( cov );
			// float hsize_invCovX = std::sqrt( inv_cov[1][1] / det_of_invcov ) * (float)SPLAT_BOUNDS;
			float hsize_invCovY = std::sqrt( inv_cov[0][0] * det_of_cov ) * (float)SPLAT_BOUNDS;

			int begY = ss_max( s.pos.y - hsize_invCovY, 0.0f );
			int endY = ss_min( s.pos.y + hsize_invCovY, image0.height() - 1.0f );
			for( int y = begY; y <= endY; y++ )
			{
				if( y < 0 || image0.height() <= y )
					continue;

				// Minimum range of x
				float vy = ( y + 0.5f ) - s.pos.y;
				float a = inv_cov[0][0];
				float b = inv_cov[1][0];
				float d = inv_cov[1][1];
				float xs[2];
				int begX = -1;
				int endX = -1;
				if( solve_quadratic( xs, a, 2.0f * b * vy, d * vy * vy - SPLAT_BOUNDS * SPLAT_BOUNDS ) )
				{
					begX = s.pos.x + xs[0];
					endX = s.pos.x + xs[1];
				}

				for( int x = begX; x <= endX; x++ )
				{
					if( x < 0 || image0.width() <= x )
						continue;

					// w as throughput
					glm::vec4 color = image1( x, y );
					float T = color.w;

					if( T < MIN_THROUGHPUT )
						continue;

					glm::vec2 p = { x + 0.5f, y + 0.5f };
					glm::vec2 v = p - s.pos;
					float d2 = glm::dot( v, inv_cov * v );
					float G = exp_approx( -0.5f * d2 );
					float alpha = G * s.opacity;

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
					glm::vec3 dC_dalpha = s.color * T - S / ( 1.0f - alpha + 1.0e-15f /* workaround zero div */ );
					glm::vec3 dL_dalpha = dL_dC * dC_dalpha;
					float dL_dalpha_rgb = dL_dalpha.x + dL_dalpha.y + dL_dalpha.z;

					// printf( "%.5f %.5f %.5f\n", S.x / ( 1.0f - alpha ), S.y / ( 1.0f - alpha ), S.z / ( 1.0f - alpha ) );
					{
						
						float a = inv_cov[0][0];
						float b = inv_cov[1][0];
						float c = inv_cov[0][1];
						float d = inv_cov[1][1];
						float dalpha_dx = 0.5f * alpha * ( 2.0f * a * v.x + ( b + c ) * v.y );
						float dalpha_dy = 0.5f * alpha * ( 2.0f * d * v.y + ( b + c ) * v.x );

						// numerical varidation x this is just for v not mu
						//float eps = 0.00001f;
						//float derivative =
						//	( s.opacity * std::expf( -0.5f * glm::dot( v + glm::vec2( eps, 0.0f ), inv_cov * ( v + glm::vec2( eps, 0.0f ) ) ) ) - s.opacity * std::expf( -0.5f * glm::dot( v, inv_cov * v ) ) ) / eps;
						//printf( "%.5f %.5f\n", dalpha_dx, -derivative );
						
						// numerical varidation y
						//float eps = 0.00001f;
						//float derivative =
						//	( s.opacity * std::expf( -0.5f * glm::dot( v + glm::vec2( 0.0f, eps ), inv_cov * ( v + glm::vec2( 0.0f, eps ) ) ) ) - s.opacity * std::expf( -0.5f * glm::dot( v, inv_cov * v ) ) ) / eps;
						//printf( "%.5f %.5f\n", dalpha_dy, -derivative );

						dSplats[i].pos.x += dL_dalpha_rgb * dalpha_dx;
						dSplats[i].pos.y += dL_dalpha_rgb * dalpha_dy;

						// vectors
						//glm::vec2 da_du = - alpha * v * glm::dot( v, s.u );
						//glm::vec2 da_dv = - alpha * v * glm::dot( v, s.v );
						//dSplats[i].u += dL_dalpha_rgb * da_du;
						//dSplats[i].v += dL_dalpha_rgb * da_dv;

						//if( trainableE )
						//{
						//	float dalpha_dE = -0.5f * alpha * glm::dot( v, v );
						//	float dalpha_dep = dalpha_dE * dE_dEp;
						//	dEp += dL_dalpha_rgb * dalpha_dep;
						//}

						// vectors v2
						
						float inv_det_of_cov_sq = 1.0f / ( det_of_cov * det_of_cov );

						float Ca = v.x * cov[1][1] - v.y * cov[0][1];
						float Cb = v.x * cov[0][1] - v.y * cov[0][0];
						float dalpha_daPrime = alpha * inv_det_of_cov_sq * 0.5f * Ca * Ca;
						float dalpha_dbPrime = -alpha * inv_det_of_cov_sq * Ca * Cb;
						float dalpha_ddPrime = alpha * inv_det_of_cov_sq * 0.5f * Cb * Cb;
						
						//float eps = 0.01f;
						//glm::mat2 cov_cp = cov;
						//cov_cp[0][0] += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, glm::inverse( cov_cp ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_daPrime, derivative );

						//float eps = 0.01f;
						//glm::mat2 cov_cp = cov;
						//cov_cp[1][1] += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, glm::inverse( cov_cp ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_ddPrime, derivative );

						//float eps = 0.001f;
						//glm::mat2 cov_cp = cov;
						//cov_cp[0][1] += eps;
						//cov_cp[1][0] += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, glm::inverse( cov_cp ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_dbPrime, derivative );

						glm::vec3 dalpha_dabcPrime = { dalpha_daPrime, dalpha_dbPrime, dalpha_ddPrime };
						glm::vec3 dabcPrime_dux = { 2.0f * s.u.x, s.u.y, 0.0f };
						float dalpha_dux = glm::dot( dalpha_dabcPrime, dabcPrime_dux );

						//float eps = 0.0001f;
						//Splat ds = s;
						//ds.u.x += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, glm::inverse( cov_of(ds) ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_dux, derivative );

						glm::vec3 dabcPrime_duy = { 0.0f, s.u.x, 2.0f * s.u.y };
						float dalpha_duy = glm::dot( dalpha_dabcPrime, dabcPrime_duy );

						//float eps = 0.0001f;
						//Splat ds = s;
						//ds.u.y += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, glm::inverse( cov_of( ds ) ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_duy, derivative );

						glm::vec3 dabcPrime_dvx = { 2.0f * s.v.x, s.v.y, 0.0f };
						float dalpha_dvx = glm::dot( dalpha_dabcPrime, dabcPrime_dvx );
						glm::vec3 dabcPrime_dvy = { 0.0f, s.v.x, 2.0f * s.v.y };
						float dalpha_dvy = glm::dot( dalpha_dabcPrime, dabcPrime_dvy );

						//float eps = 0.0001f;
						//Splat ds = s;
						//ds.v.x += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, glm::inverse( cov_of(ds) ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_dvx, derivative );
						//float eps = 0.0001f;
						//Splat ds = s;
						//ds.v.y += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, glm::inverse( cov_of( ds ) ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_dvy, derivative );

						dSplats[i].u.x += dL_dalpha_rgb * dalpha_dux;
						dSplats[i].u.y += dL_dalpha_rgb * dalpha_duy;
						dSplats[i].v.x += dL_dalpha_rgb * dalpha_dvx;
						dSplats[i].v.y += dL_dalpha_rgb * dalpha_dvy;

						 // numerical varidation
						//float prevEp = Ep; 
						//float eps = 0.001f;
						//Splat ds = s;
						//Ep += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, inv_cov_of( ds ) * v ) ) - alpha ) / eps;
						//Ep = prevEp;
						//printf( "%f %f\n", dalpha_dep, derivative );

						// rotation
						//float dalpha_dsx =
						//	alpha / ( s.sx * s.sx * s.sx ) *
						//	glm::dot( glm::vec3( cosTheta * cosTheta, 2.0f * sinTheta * cosTheta, sinTheta * sinTheta ), glm::vec3( v.x * v.x, v.x * v.y, v.y * v.y ) );
						//float dalpha_dsy =
						//	alpha / ( s.sy * s.sy * s.sy ) *
						//	glm::dot( glm::vec3( sinTheta * sinTheta, -2.0f * sinTheta * cosTheta, cosTheta * cosTheta ), glm::vec3( v.x * v.x, v.x * v.y, v.y * v.y ) );

						// numerical varidation
						//float eps = 0.0001f; 
						//Splat ds = s;
						//ds.sx += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, glm::inverse( cov_of( ds ) ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_dsx, derivative );

						//float eps = 0.0001f;
						//Splat ds = s;
						//ds.sy += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, glm::inverse( cov_of( ds ) ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_dsy, derivative );

						// scale
						//dSplats[i].sx += dL_dalpha_rgb * dalpha_dsx;
						//dSplats[i].sy += dL_dalpha_rgb * dalpha_dsy;

						//float dalpha_dtheta =
						//	alpha *
						//	( s.sx * s.sx - s.sy * s.sy ) / ( s.sx * s.sx * s.sy * s.sy ) *
						//	( ( cosTheta * cosTheta - sinTheta * sinTheta ) * v.x * v.y - sinTheta * cosTheta * ( v.x * v.x - v.y * v.y ) );

						//dSplats[i].rot += ( dL_dalpha.x + dL_dalpha.y + dL_dalpha.z ) * dalpha_dtheta;


						// numerical varidation
						//float eps = 0.001f;
						//Splat ds = s;
						//ds.rot += eps;
						//float derivative = ( s.opacity * exp_approx( -0.5f * glm::dot( v, glm::inverse( cov_of( ds ) ) * v ) ) - alpha ) / eps;
						//printf( "%f %f\n", dalpha_dtheta, derivative );

						//float derivative = ( glm::inverse( cov_of( ds ) )[0][0] - a ) / eps;
						//printf( "%f %f\n", da_dtheta, derivative );

						//float derivative = ( glm::inverse( cov_of( ds ) )[1][0] - b ) / eps;
						//printf( "%f %f\n", db_dtheta, derivative );

						//float derivative = ( glm::inverse( cov_of( ds ) )[1][1] - d ) / eps;
						//printf( "%f %f\n", dd_dtheta, derivative );

						float dalpha_do = G;
						dSplats[i].opacity += dL_dalpha_rgb * dalpha_do;
					}

					color.w *= ( 1.0f - alpha );

					image1( x, y ) = color;
				}
			}
		}

		// optimize
		float trainingRate = 0.05f;

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


			splats[i].u.x = splatAdams[i].u[0].optimize( splats[i].u.x, dSplats[i].u.x, trainingRate, beta1t, beta2t );
			splats[i].u.y = splatAdams[i].u[1].optimize( splats[i].u.y, dSplats[i].u.y, trainingRate, beta1t, beta2t );
			splats[i].v.x = splatAdams[i].v[0].optimize( splats[i].v.x, dSplats[i].v.x, trainingRate, beta1t, beta2t );
			splats[i].v.y = splatAdams[i].v[1].optimize( splats[i].v.y, dSplats[i].v.y, trainingRate, beta1t, beta2t );

			//splats[i].sx = splatAdams[i].sx.optimize( splats[i].sx, dSplats[i].sx, trainingRate, beta1t, beta2t );
			//splats[i].sy = splatAdams[i].sy.optimize( splats[i].sy, dSplats[i].sy, trainingRate, beta1t, beta2t );

			//splats[i].rot = splatAdams[i].rot.optimize( splats[i].rot, dSplats[i].rot, trainingRate, beta1t, beta2t );

			if( optimizeOpacity )
			{
				splats[i].opacity = splatAdams[i].opacity.optimize( splats[i].opacity, dSplats[i].opacity, trainingRate, beta1t, beta2t );
			}

			// constraints
			splats[i].pos.x = glm::clamp( splats[i].pos.x, 0.0f, (float)imageRef.width() - 1 );
			splats[i].pos.y = glm::clamp( splats[i].pos.y, 0.0f, (float)imageRef.height() - 1 );

			//float crossVal = splats[i].u.x * splats[i].v.y - splats[i].u.y * splats[i].v.x;
			//if (glm::abs(crossVal) < 0.01f)
			//{
			//	printf( "crossVal %f\n", crossVal );
			//}

			//auto new_cov = cov_of( splats[i] );
			//float d = glm::determinant( new_cov );
			//if (d == 0.0f)
			//{
			//	printf( "a\n" );
			//}
			//if (isfinite(dSplats[i].pos.x) == false)
			//{
			//	printf( "a\n" );
			//}
			//splats[i].sx = glm::clamp( splats[i].sx, 1.0f, 1024.0f );
			//splats[i].sy = glm::clamp( splats[i].sy, 1.0f, 1024.0f );

			//auto inv_cov = inv_cov_of( splats[i] );
			//float det_of_invcov;
			//float lambda0_inv;
			//float lambda1_inv;
			//eignValues( &lambda0_inv, &lambda1_inv, &det_of_invcov, inv_cov );

			//const float maxPixels = 32.f;
			//const float clampVal = 1.0f / ( maxPixels * maxPixels );
			//if( lambda0_inv < clampVal || lambda1_inv < clampVal )
			//{
			//	glm::vec2 e0;
			//	glm::vec2 e1;
			//	eigenVectors_of_symmetric( &e0, &e1, inv_cov, lambda0_inv );
			//	splats[i].u = e0 * std::sqrtf( ss_max( lambda0_inv, clampVal ) );
			//	splats[i].v = e1 * std::sqrtf( ss_max( lambda1_inv, clampVal ) );
			//}
			//float len0 = std::sqrtf( 1.0f / lambda0_inv ); // L
			//float len1 = std::sqrtf( 1.0f / lambda1_inv ); // S
			//const float maxMul = 8.0f;
			//if( len0 * maxMul < len1 )
			//{
			//	glm::vec2 e0;
			//	glm::vec2 e1;
			//	eigenVectors_of_symmetric( &e0, &e1, inv_cov, lambda0_inv );
			//	splats[i].u = e0 * std::sqrtf( lambda0_inv );
			//	splats[i].v = e1 / ss_min( len1, len0 * maxMul );
			//}

			splats[i].color = glm::clamp( splats[i].color, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } );

			splats[i].opacity = glm::clamp( splats[i].opacity, 0.1f, 1.0f );
		}

#if 1
		for (int i = 0; i < splats.size(); i++)
		{
			auto s = splats[i];
			if( isfinite( s.color.x ) == false )
			{
				abort();
			}
			if( isfinite( s.color.y ) == false )
			{
				abort();
			}
			if( isfinite( s.color.z ) == false )
			{
				abort();
			}
			//if( isfinite( s.sx ) == false )
			//{
			//	abort();
			//}
			//if( isfinite( s.sy ) == false )
			//{
			//	abort();
			//}
			//if( isfinite( s.rot ) == false )
			//{
			//	abort();
			//}
			if( isfinite( s.pos.x ) == false )
			{
				abort();
			}
		}
#endif

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

		printf( "%d itr, mse %.4f\n", iterations, mse );

		iterations++;

		const int NMeasure = 2048;
		static std::vector<float> mseList;
		mseList.reserve( NMeasure );
		if( mseList.size() < NMeasure )
		{
			mseList.push_back( mse );
			if( mseList.size() == NMeasure )
			{
				FILE* fp = fopen( "measure_vec.csv", "w" );
				fprintf( fp, "mse\n" );
				for( int i = 0; i < mseList.size() ; i++)
				{
					fprintf( fp, "%.10f\n", mseList[i] );
				}
				fclose( fp );
			}
		}

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowPos({ 20, 20 }, ImGuiCond_Once);
        ImGui::SetNextWindowSize({ 600, 1200 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
		ImGui::Text( "%d itr, mse %.4f", iterations, mse );
		ImGui::Text( "%d splats", NSplat );
		if( ImGui::Button( "2x splat" ) )
		{
			NSplat *= 2;
			init();
		}
		if( ImGui::Button( "1/2 splat" ) )
		{
			NSplat = ss_max( NSplat / 2, 1 );
			init();
		}
		static int viewScale = 2;
		ImGui::InputInt( "viewScale", &viewScale );

		ImGui::Checkbox( "Optimize opacity", &optimizeOpacity );
		ImGui::Checkbox( "Show splat info", &showSplatInfo );

		if( ImGui::Button( "Restart" ) )
		{
			init();
		}

		//if (ImGui::Button("d"))
		//{
		//	for (int i = 0; i < splats.size(); i++)
		//	{
		//		// printf( "%f, %f, %f{%f,%f} {%f,%f}\n", splats[i].color.x, splats[i].color.y, splats[i].color.z, splats[i].u.x, splats[i].u.y, splats[i].v.x, splats[i].v.y );
		//		float crossVal = splats[i].u.x * splats[i].v.y - splats[i].u.y * splats[i].v.x;
		//		printf( "%f, %f, %f{%f,%f} {%f,%f} %f\n", splats[i].color.x, splats[i].color.y, splats[i].color.z, splats[i].u.x, splats[i].u.y, splats[i].v.x, splats[i].v.y, crossVal );
		//	}
		//}
		
		viewScale = ss_max( viewScale, 1 );

		//ImGui::SliderFloat( "sx", &splat_sx, 0, 64 );
		//ImGui::SliderFloat( "sy", &splat_sy, 0, 64 );
		//ImGui::SliderFloat( "rot", &splat_rot, -glm::pi<float>(), glm::pi<float>() );
   
        ImGui::Image( textureRef, ImVec2( textureRef->width() * viewScale, textureRef->height() * viewScale ) );
		ImGui::Image( tex0, ImVec2( tex0->width() * viewScale, tex0->height() * viewScale ) );

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
