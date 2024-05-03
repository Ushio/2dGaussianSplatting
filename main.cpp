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
    Adam radius;
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
	float cosTheta = std::cosf( splat.rot );
	float sinTheta = std::sinf( splat.rot );
	float lambda0 = splat.sx * splat.sx;
	float lambda1 = splat.sy * splat.sy;
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

    int NSplat = 32;
    std::vector<Splat> splats(NSplat);

    for( int i = 0; i < splats.size(); i++ )
	{
		glm::vec3 r0 = glm::vec3( pcg3d( { i, 0, 0xFFFFFFFF } ) ) / glm::vec3( 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );
		glm::vec3 r1 = glm::vec3( pcg3d( { i, 1, 0xFFFFFFFF } ) ) / glm::vec3( 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );

		Splat s;
		s.pos.x = glm::mix( r0.x, (float)imageRef.width() - 1, r0.x );
		s.pos.y = glm::mix( r0.y, (float)imageRef.height() - 1, r0.y );
		s.sx = glm::mix( 4.0f, 8.0f, r1.x );
		s.sy = glm::mix( 4.0f, 8.0f, r1.y );
		s.rot = glm::pi<float>() * 2.0f * r1.z;
		s.color = { 0.5f, 0.5f, 0.5f };
		splats[i] = s;
	}

    float beta1t = 1.0f;
    float beta2t = 1.0f;
    std::vector<SplatAdam> splatAdams(splats.size());


    ITexture* tex0 = CreateTexture();
    Image2DRGBA32 image0;
    image0.allocate(imageRef.width(), imageRef.height());

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

            float r = ss_max( sqrt_of_lambda0, sqrt_of_lambda1 ) * 2.0f;
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
					
					glm::vec2 p = { x + 0.5f, y + 0.5f };
					glm::vec2 v = p - s.pos;
					float alpha = std::expf( -0.5f * glm::dot( v, inv_cov * v ) );

					color.x += color.w * s.color.x * alpha;
					color.y += color.w * s.color.y * alpha;
					color.z += color.w * s.color.z * alpha;

                    color.w *= ( 1.0f - alpha );

					image0( x, y ) = color;
				}
			}
		}

        // backward


        image0 = image0.map( []( const glm::vec4& c ) { return glm::vec4( c.x, c.y, c.z, 1.0f ); } );

        tex0->upload(image0);

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowPos({ 20, 20 }, ImGuiCond_Once);
        ImGui::SetNextWindowSize({ 600, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
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
