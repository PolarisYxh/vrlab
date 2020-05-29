#version 400 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
 //new
uniform Mat{
	vec4 aAmbient;
	vec4 aDiffuse;
	vec4 aSpecular;
};


struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};

uniform vec3 viewPos;
uniform Light light;
uniform float shininess;//

out vec2 TexCoord;
out vec3 LightIntensity;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
 
void main(){
//new
  vec3 FragPos = vec3( model * vec4(aPos, 1.0));
  vec3 Normal = mat3(transpose(inverse(model))) * aNormal;
  //new for light
   	// ambient 1 1 1*0.1
    vec3 ambient = light.ambient * aAmbient.rgb;
  	
    // diffuse 1 1 1 *1 1 1
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse =light.diffuse * diff *aDiffuse.rgb;  
      
    // attenuation�Ȳ�����˥��
   // float distance    = length(light.position - FragPos);
    //float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
	// specular  1 1 1 *0.5 0.5 0.5
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = light.specular * spec *   aSpecular.rgb;  
	
    //ambient  *= attenuation;  ��������ע�͵�
    //diffuse   *= attenuation; 
    //specular *= attenuation; 
	  
    LightIntensity = ambient + diffuse +specular;//
  TexCoord = aTexCoord;
  gl_Position = projection * view * model * vec4(aPos, 1.0f);
}