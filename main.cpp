#include "pr.hpp"
#include <iostream>
#include <memory>

#include <intrin.h>

#define SPLAT_BOUNDS 3.0f
#define ALPHA_THRESHOLD ( 1.0f / 256.0f )

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
	//return expf( x ); // use this for numerical varidation

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
	float sx;
	float sy;
	float rot;
    glm::vec3 color;
	float opacity;
};


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
	Adam sx;
	Adam sy;
	Adam rot;
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
glm::mat2 cov_of( const Splat& splat )
{
	float theta = splat.rot;
	float sx = splat.sx;
	float sy = splat.sy;

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


// Numerical Differenciate
#define POS_PURB 0.1f
#define S_PURB 0.2f
#define COLOR_PURB 0.01f
#define ROT_PURB 0.02f
#define OPACITY_PURB 0.01f

enum
{
	SIGNBIT_POS_X = 0,
	SIGNBIT_POS_Y,
	SIGNBIT_SX,
	SIGNBIT_SY,
	SIGNBIT_ROT,
	SIGNBIT_COL_R,
	SIGNBIT_COL_G,
	SIGNBIT_COL_B,
	SIGNBIT_OPACITY,
};

bool bitAt( uint32_t u, uint32_t i )
{
	return u & ( 1u << i );
}

// 0: +1, 1: -1
float signAt( uint32_t u, uint32_t i )
{
	return bitAt( u, i ) ? -1.0f : 1.0f;
}

uint32_t splatRng( uint32_t i, uint32_t perturbIdx )
{
	return pcg( i + pcg( perturbIdx ) );
}


void drawSplats( pr::Image2DRGBA32* image, std::vector<int>* splatIndices, const std::vector<Splat>& splats, const uint32_t* splatRNGs, float sign )
{
	int w = image->width();
	int h = image->height();
	for( int i = 0; i < splats.size(); i++ )
	{
		Splat s = splats[i];

		// Apply perturb
		uint32_t rng = splatRNGs[i];
		s.pos.x += sign * POS_PURB * signAt( rng, SIGNBIT_POS_X );
		s.pos.y += sign * POS_PURB * signAt( rng, SIGNBIT_POS_Y );

		s.sx += sign * S_PURB * signAt( rng, SIGNBIT_SX );
		s.sy += sign * S_PURB * signAt( rng, SIGNBIT_SY );

		s.rot += sign * ROT_PURB * signAt( rng, SIGNBIT_ROT );

		s.color.x += sign * COLOR_PURB * signAt( rng, SIGNBIT_COL_R );
		s.color.y += sign * COLOR_PURB * signAt( rng, SIGNBIT_COL_G );
		s.color.z += sign * COLOR_PURB * signAt( rng, SIGNBIT_COL_B );

		// s.opacity += sign * OPACITY_PURB * signAt( rng, SIGNBIT_OPACITY );

		// constraints
		s.sx = glm::clamp( s.sx, 1.0f, 1024.0f );
		s.sy = glm::clamp( s.sy, 1.0f, 1024.0f );
		s.color = glm::clamp( s.color, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } );

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

		float r = ss_max( sqrt_of_lambda0, sqrt_of_lambda1 ) * SPLAT_BOUNDS;
		int begX = s.pos.x - r;
		int endX = s.pos.x + r;
		int begY = s.pos.y - r;
		int endY = s.pos.y + r;
		for( int y = begY; y <= endY; y++ )
		{
			if( y < 0 || h <= y )
				continue;

			for( int x = begX; x <= endX; x++ )
			{
				if( x < 0 || w <= x )
					continue;

				// w as throughput
				glm::vec4 color = (*image)( x, y );
				float T = color.w;
				if( T < 1.0f / 256.0f )
					continue;

				glm::vec2 p = { x + 0.5f, y + 0.5f };
				glm::vec2 v = p - s.pos;
				
				float d2 = glm::dot( v, inv_cov * v );
				if( SPLAT_BOUNDS * SPLAT_BOUNDS < d2 )
					continue;

				float alpha = exp_approx( -0.5f * d2 ) * s.opacity;
				if( alpha < ALPHA_THRESHOLD )
					continue;

				color.x += T * s.color.x * alpha;
				color.y += T * s.color.y * alpha;
				color.z += T * s.color.z * alpha;

				color.w *= ( 1.0f - alpha );

				( *image )( x, y ) = color;

				if( splatIndices )
				{
					splatIndices[y * w + x].push_back( i );
				}
			}
		}
	}
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
		image.load( "squirrel_cls_mini.jpg" );
		// image.load( "squirrel_cls_micro.jpg" );
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
	// int NSplat = 512;
	std::vector<Splat> splats( NSplat );
	
    for( int i = 0; i < splats.size(); i++ )
	{
		glm::vec3 r0 = glm::vec3( pcg3d( { i, 0, 0xFFFFFFFF } ) ) / glm::vec3( 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );
		glm::vec3 r1 = glm::vec3( pcg3d( { i, 1, 0xFFFFFFFF } ) ) / glm::vec3( 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );

		Splat s;
		s.pos.x = glm::mix( r0.x, (float)imageRef.width() - 1, r0.x );
		s.pos.y = glm::mix( r0.y, (float)imageRef.height() - 1, r0.y );
		s.sx = glm::mix( 6.0f, 10.0f, r1.x );
		s.sy = glm::mix( 6.0f, 10.0f, r1.y );
		// s.sx = glm::mix( 3.0f, 6.0f, r1.x );
		// s.sy = glm::mix( 3.0f, 6.0f, r1.y );
		//s.sx = 8;
		//s.sy = 8;
		s.rot = glm::pi<float>() * r1.z;
		s.color = { 0.5f, 0.5f, 0.5f };
		s.opacity = 1.0f;
		splats[i] = s;
	}

    float beta1t = 1.0f;
    float beta2t = 1.0f;
    std::vector<SplatAdam> splatAdams(splats.size());

	beta1t = 1.0f;
	beta2t = 1.0f;
	splatAdams.clear();
	splatAdams.resize( NSplat );

	int iterations = 0;


    ITexture* tex0 = CreateTexture();
	Image2DRGBA32 image0;
	image0.allocate( imageRef.width(), imageRef.height() );

	Image2DRGBA32 image1;
	image1.allocate( imageRef.width(), imageRef.height() );

	bool showSplatInfo = false;
	bool optimizeOpacity = false;

	ITexture* perturb0Tex = CreateTexture();
	Image2DRGBA32 perturb0;
	Image2DRGBA32 perturb1;
	perturb0.allocate( imageRef.width(), imageRef.height() );
	perturb1.allocate( imageRef.width(), imageRef.height() );

    std::vector<std::vector<int>> indices0(imageRef.width() * imageRef.height());

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

	enum
	{
		OPTIMIZER_ANALYTIC = 0,
		OPTIMIZER_NUMERICAL,
	};
	int optimizerMode = OPTIMIZER_NUMERICAL;

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

			// visuallize
			{
				glm::vec2 axis0 = eigen0 * sqrt_of_lambda0;
				glm::vec2 axis1 = eigen1 * sqrt_of_lambda1;

				//Draw axis
				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ), { 255, 255, 255 } );
				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ) + glm::vec3( axis0.x, -axis0.y, 0 ), { 255, 255, 255 } );

				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ), { 255, 255, 255 } );
				PrimVertex( glm::vec3( s.pos.x, -s.pos.y, 0 ) + glm::vec3( axis1.x, -axis1.y, 0 ), { 230, 230, 230 } );

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

				if (showSplatInfo)
				{
					char op[128];
					sprintf( op, "o=%.2f, c=(%.2f, %.2f, %.2f)", s.opacity, s.color.x, s.color.y, s.color.z );
					DrawText( glm::vec3( s.pos.x, -s.pos.y, 0 ), op, 12 );
				}
			}

            float r = ss_max( sqrt_of_lambda0, sqrt_of_lambda1 ) * SPLAT_BOUNDS;
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

					float alpha = exp_approx( -0.5f * glm::dot( v, inv_cov * v ) ) * s.opacity;
					if( alpha < ALPHA_THRESHOLD )
						continue;

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
		if( optimizerMode == OPTIMIZER_ANALYTIC )
		{
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

				float theta = s.rot;
				float cosTheta = std::cosf( theta );
				float sinTheta = std::sinf( theta );

				float r = ss_max( sqrt_of_lambda0, sqrt_of_lambda1 ) * SPLAT_BOUNDS;
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
						float G = exp_approx( -0.5f * glm::dot( v, inv_cov * v ) );
						float alpha = G * s.opacity;
						if( alpha < ALPHA_THRESHOLD )
							continue;

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

							float dalpha_dsx =
								alpha / ( s.sx * s.sx * s.sx ) *
								glm::dot( glm::vec3( cosTheta * cosTheta, 2.0f * sinTheta * cosTheta, sinTheta * sinTheta ), glm::vec3( v.x * v.x, v.x * v.y, v.y * v.y ) );
							float dalpha_dsy =
								alpha / ( s.sy * s.sy * s.sy ) *
								glm::dot( glm::vec3( sinTheta * sinTheta, -2.0f * sinTheta * cosTheta, cosTheta * cosTheta ), glm::vec3( v.x * v.x, v.x * v.y, v.y * v.y ) );

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

							dSplats[i].sx += dL_dalpha_rgb * dalpha_dsx;
							dSplats[i].sy += dL_dalpha_rgb * dalpha_dsy;

							float dalpha_dtheta =
								alpha *
								( s.sx * s.sx - s.sy * s.sy ) / ( s.sx * s.sx * s.sy * s.sy ) *
								( ( cosTheta * cosTheta - sinTheta * sinTheta ) * v.x * v.y - sinTheta * cosTheta * ( v.x * v.x - v.y * v.y ) );

							dSplats[i].rot += ( dL_dalpha.x + dL_dalpha.y + dL_dalpha.z ) * dalpha_dtheta;

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

				splats[i].sx = splatAdams[i].sx.optimize( splats[i].sx, dSplats[i].sx, trainingRate, beta1t, beta2t );
				splats[i].sy = splatAdams[i].sy.optimize( splats[i].sy, dSplats[i].sy, trainingRate, beta1t, beta2t );

				splats[i].rot = splatAdams[i].rot.optimize( splats[i].rot, dSplats[i].rot, trainingRate, beta1t, beta2t );

				if( optimizeOpacity )
				{
					splats[i].opacity = splatAdams[i].opacity.optimize( splats[i].opacity, dSplats[i].opacity, trainingRate, beta1t, beta2t );
				}

				// constraints
				splats[i].pos.x = glm::clamp( splats[i].pos.x, 0.0f, (float)imageRef.width() - 1 );
				splats[i].pos.y = glm::clamp( splats[i].pos.y, 0.0f, (float)imageRef.height() - 1 );

				splats[i].sx = glm::clamp( splats[i].sx, 1.0f, 1024.0f );
				splats[i].sy = glm::clamp( splats[i].sy, 1.0f, 1024.0f );

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
				if( isfinite( s.sx ) == false )
				{
					abort();
				}
				if( isfinite( s.sy ) == false )
				{
					abort();
				}
				if( isfinite( s.rot ) == false )
				{
					abort();
				}
				if( isfinite( s.pos.x ) == false )
				{
					abort();
				}
			}
	#endif

			iterations++;
		}

		if( optimizerMode == OPTIMIZER_NUMERICAL )
		{
			static int perturbIdx = 0;

			std::vector<Splat> dSplats( splats.size() );
			std::vector<uint32_t> splatRNGs( splats.size() );

			int w = imageRef.width();
			int h = imageRef.height();

			for( int i = 0; i < 32; i++ )
			{
				// update rngs for each sub iteration
				for( int j = 0; j < splatRNGs.size(); j++ )
				{
					splatRNGs[j] = splatRng( j, perturbIdx );
				}

				std::fill( perturb0.data(), perturb0.data() + perturb0.width() * perturb0.height(), glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );
				std::fill( perturb1.data(), perturb1.data() + perturb1.width() * perturb1.height(), glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );

				for( int i = 0; i < indices0.size(); i++ )
					indices0[i].clear();

				drawSplats( &perturb0, indices0.data(), splats, splatRNGs.data(), +1.0f );
				drawSplats( &perturb1, nullptr        , splats, splatRNGs.data(), -1.0f );

				for( int y = 0; y < h; y++ )
				{
					for( int x = 0; x < w; x++ )
					{
						glm::vec3 d0 = imageRef( x, y ) - perturb0( x, y );
						glm::vec3 d1 = imageRef( x, y ) - perturb1( x, y );
						float fwh0 = lengthSquared( d0 );
						float fwh1 = lengthSquared( d1 );
						float df = fwh0 - fwh1;

						for( int i : indices0[y * imageRef.width() + x] )
						{
							uint32_t r = splatRNGs[i];

							dSplats[i].pos.x += df * signAt( r, SIGNBIT_POS_X );
							dSplats[i].pos.y += df * signAt( r, SIGNBIT_POS_Y );

							dSplats[i].sx += df * signAt( r, SIGNBIT_SX );
							dSplats[i].sy += df * signAt( r, SIGNBIT_SY );

							dSplats[i].rot += df * signAt( r, SIGNBIT_ROT );

							dSplats[i].color.x += df * signAt( r, SIGNBIT_COL_R );
							dSplats[i].color.y += df * signAt( r, SIGNBIT_COL_G );
							dSplats[i].color.z += df * signAt( r, SIGNBIT_COL_B );

							//dSplats[i].opacity += df * signAt( r, SIGNBIT_OPACITY );
						}
					}
				}

				perturbIdx++;
			}

			// gradient decent
			beta1t *= ADAM_BETA1;
			beta2t *= ADAM_BETA2;

			float trainingScale = 1.0f;
			// based on the paper, no div by N, but use purb amount as learning rate.

			for( int i = 0; i < splats.size(); i++ )
			{
				splats[i].color.x = splatAdams[i].color[0].optimize( splats[i].color.x, dSplats[i].color.x, COLOR_PURB * trainingScale, beta1t, beta2t );
				splats[i].color.y = splatAdams[i].color[1].optimize( splats[i].color.y, dSplats[i].color.y, COLOR_PURB * trainingScale, beta1t, beta2t );
				splats[i].color.z = splatAdams[i].color[2].optimize( splats[i].color.z, dSplats[i].color.z, COLOR_PURB * trainingScale, beta1t, beta2t );

				splats[i].pos.x = splatAdams[i].pos[0].optimize( splats[i].pos.x, dSplats[i].pos.x, POS_PURB * trainingScale, beta1t, beta2t );
				splats[i].pos.y = splatAdams[i].pos[1].optimize( splats[i].pos.y, dSplats[i].pos.y, POS_PURB * trainingScale, beta1t, beta2t );

				splats[i].sx = splatAdams[i].sx.optimize( splats[i].sx, dSplats[i].sx, S_PURB * trainingScale, beta1t, beta2t );
				splats[i].sy = splatAdams[i].sy.optimize( splats[i].sy, dSplats[i].sy, S_PURB * trainingScale, beta1t, beta2t );

				splats[i].rot = splatAdams[i].rot.optimize( splats[i].rot, dSplats[i].rot, ROT_PURB * trainingScale, beta1t, beta2t );

				if( optimizeOpacity )
				{
					splats[i].opacity = splatAdams[i].opacity.optimize( splats[i].opacity, dSplats[i].opacity, OPACITY_PURB * trainingScale, beta1t, beta2t );
				}

				// constraints
				splats[i].pos.x = glm::clamp( splats[i].pos.x, 0.0f, (float)imageRef.width() - 1 );
				splats[i].pos.y = glm::clamp( splats[i].pos.y, 0.0f, (float)imageRef.height() - 1 );

				splats[i].sx = glm::clamp( splats[i].sx, 1.0f, 1024.0f );
				splats[i].sy = glm::clamp( splats[i].sy, 1.0f, 1024.0f );

				//float m = sqrtf( splats[i].sx * splats[i].sy );
				//m = glm::clamp( m, 4.0f, 16.0f );
				//splats[i].sx = m;
				//splats[i].sy = m;

				splats[i].color = glm::clamp( splats[i].color, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } );

				splats[i].opacity = glm::clamp( splats[i].opacity, 0.1f, 1.0f );

			}
#if 1
			for( int i = 0; i < splats.size(); i++ )
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
				if( isfinite( s.sx ) == false )
				{
					abort();
				}
				if( isfinite( s.sy ) == false )
				{
					abort();
				}
				if( isfinite( s.rot ) == false )
				{
					abort();
				}
				if( isfinite( s.pos.x ) == false )
				{
					abort();
				}
			}
#endif

			for( int i = 0; i < image0.width() * image0.height(); i++ )
			{
				perturb0.data()[i].w = 1.0f;
			}
			perturb0Tex->upload( perturb0 );

			iterations++;
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

		printf( "%d itr, mse %.4f\n", iterations, mse );

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowPos({ 20, 20 }, ImGuiCond_Once);
        ImGui::SetNextWindowSize({ 600, 1000 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
		ImGui::Text( "%d itr, mse %.4f", iterations, mse );
		ImGui::Text( "%d splats", NSplat );
		static int viewScale = 2;
		ImGui::InputInt( "viewScale", &viewScale );

		ImGui::Checkbox( "Optimize opacity", &optimizeOpacity );
		ImGui::Checkbox( "Show splat info", &showSplatInfo );

		ImGui::RadioButton( "Optimizer Analytic", &optimizerMode, OPTIMIZER_ANALYTIC );
		ImGui::RadioButton( "Optimizer Numerical", &optimizerMode, OPTIMIZER_NUMERICAL );
		
		viewScale = ss_max( viewScale, 1 );

		//ImGui::SliderFloat( "sx", &splat_sx, 0, 64 );
		//ImGui::SliderFloat( "sy", &splat_sy, 0, 64 );
		//ImGui::SliderFloat( "rot", &splat_rot, -glm::pi<float>(), glm::pi<float>() );
   
        ImGui::Image( textureRef, ImVec2( textureRef->width() * viewScale, textureRef->height() * viewScale ) );
		ImGui::Image( tex0, ImVec2( tex0->width() * viewScale, tex0->height() * viewScale ) );
		ImGui::Image( perturb0Tex, ImVec2( perturb0Tex->width() * viewScale, perturb0Tex->height() * viewScale ) );
		
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
