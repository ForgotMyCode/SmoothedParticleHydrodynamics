#include <Window.h>
#include <Scene.h>
#include <Guard.h>
#include <Shader.h>
#include <Loader.h>
#include <Geometry.h>
#include <iterator>
#include <Mesh.h>
#include <Shapes.h>
#include <Texture.h>
#include <Simulation/ParticleSimulatingMesh.h>

#include <glad/glad.h>

int main() {
	Window window{};

	Scene testScene{};

	CompiledShaderUnit vs = shaderCompiler::compileShaderFromSource(loader::loadTextFromFile("./Shaders/simplevs.vert"), ShaderType::VertexShader);
	CompiledShaderUnit fs = shaderCompiler::compileShaderFromSource(loader::loadTextFromFile("./Shaders/simplefs.frag"), ShaderType::FragmentShader);

	Shader shader({ vs, fs });

	window.GetResourceManager().AddShader("SimpleShader", &shader);

	float vertices[] = {
		-0.5f, -0.5f, 0.0f,
		 0.5f, -0.5f, 0.0f,
		 0.0f,  0.5f, 0.0f
	};

	Geometry geometry({ 3 });

	auto& buffer = geometry.BeginChangingGeometry();

	buffer.insert(buffer.end(), std::begin(vertices), std::end(vertices));

	geometry.FinishChangingGeometry();

	Mesh mesh(&geometry, &shader);

	mesh.SetScale(1.f);

	testScene.AddObject(&mesh);

	Geometry box({ 3 }, true, Geometry::EPrimitive::Line);

	auto& boxBuffer = box.BeginChangingGeometry();

	auto boxVertices = shapes::axisAlignedLinedBox<
		config::simulation::boundingBox::minX,
		config::simulation::boundingBox::maxX,
		config::simulation::boundingBox::minY,
		config::simulation::boundingBox::maxY,
		config::simulation::boundingBox::minZ,
		config::simulation::boundingBox::maxZ
	>();

	boxBuffer.insert(boxBuffer.end(), boxVertices.begin(), boxVertices.end());

	box.FinishChangingGeometry();

	Mesh boxMesh(&box, &shader);

	boxMesh.SetScale(1.f);

	testScene.AddObject(&boxMesh);

	auto image = loader::loadImage("Textures/waterparticle_v3.png");

	Texture texture(image);

	window.GetResourceManager().AddTexture("water particle", &texture);

	Geometry particleGeometry({ 2, 2 });

	auto& particleBuffer = particleGeometry.BeginChangingGeometry();

	auto particleData = shapes::texturedUniquad<-0.1f, 0.1f, -0.1f, 0.1f>();

	particleBuffer.insert(particleBuffer.end(), particleData.begin(), particleData.end());

	particleGeometry.FinishChangingGeometry();

	CompiledShaderUnit particleVs = shaderCompiler::compileShaderFromSource(loader::loadTextFromFile("./Shaders/projecting2d.vert"), ShaderType::VertexShader);
	CompiledShaderUnit particleFs = shaderCompiler::compileShaderFromSource(loader::loadTextFromFile("./Shaders/texturingfs.frag"), ShaderType::FragmentShader);

	Shader particleShader({ particleVs, particleFs });

	TexturedMesh texturedParticle(&particleGeometry, &particleShader, &texture);

	texturedParticle.SetLocation(glm::vec3(0.0f, 1.0f, 0.0f));

	//testScene.AddObject(&texturedParticle);

	ParticleSimulatingMesh particleMesh(&particleGeometry, &particleShader, &texture);

	testScene.AddObject(&particleMesh);

	window.SetActiveScene(&testScene);

	window.MainLoop();

	return 0;
}