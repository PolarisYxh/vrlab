#version 330 core
out vec4 FragColor;
 
//new for light
struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};
in vec3 FragPos;  
in vec3 Normal;  
//��Mtl�ж�ȡ������
//Material
in vec4 Ambient;
in vec4 Diffuse;
in vec4 Specular;

uniform vec3 viewPos;
uniform Light light;
uniform float shininess;//

in vec2 TexCoord;
 
uniform sampler2D texture_diffuse1;//û������
uniform sampler2D texture_height1;
 
void main()
{
//new for light
   	// ambient 1 1 1*0.1
    vec3 ambient = light.ambient * Ambient.rgb;
  	
    // diffuse 1 1 1 *1 1 1
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse =light.diffuse * diff *Diffuse.rgb;  
      
    // attenuation�Ȳ�����˥��
   // float distance    = length(light.position - FragPos);
    //float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
	// specular  1 1 1 *0.5 0.5 0.5
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = light.specular * spec *   Specular.rgb;  
	
    //ambient  *= attenuation;  ��������ע�͵�
    //diffuse   *= attenuation; 
    //specular *= attenuation; 
	  
    vec3 result = ambient + diffuse +specular;//
    vec4 objectcolor=texture2D(texture_diffuse1, TexCoord);
    //objectcolor=vec4(0.5,0.5,0.5,1);
    FragColor =objectcolor*vec4(result ,1.0);
    //FragColor=vec4(LightIntensity ,1.0);
    //FragColor = mix(texture2D(texture_diffuse1, TexCoord), texture2D(texture_height1, TexCoord), 0.2f);
      //FragColor = texture2D(texture_diffuse1, TexCoord);
}