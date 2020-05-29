#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
 //new
uniform Mat{
	vec4 aAmbient;
	vec4 aDiffuse;
	vec4 aSpecular;
};
out vec3 FragPos;
out vec3 Normal;
out vec4 Ambient;
out vec4 Diffuse;
out vec4 Specular;//

out vec2 TexCoord;
 
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
 
void main(){
//new
  FragPos = vec3( model * vec4(aPos, 1.0));
  Normal = mat3(transpose(inverse(model))) * aNormal;
  Ambient = aAmbient;
  Diffuse = aDiffuse;
  Specular = aSpecular;//
  TexCoord = aTexCoord;
  gl_Position = projection * view * model * vec4(aPos, 1.0f);
}