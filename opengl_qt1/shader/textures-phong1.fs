#version 400 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 LightIntensity;
uniform sampler2D texture_diffuse1;//û������
uniform sampler2D texture_height1;
 
void main()
{

    vec4 objectcolor=texture2D(texture_diffuse1, TexCoord);
    //objectcolor=vec4(0.5,0.5,0.5,1);
    FragColor =objectcolor*vec4(LightIntensity ,1.0);
    //FragColor=vec4(LightIntensity ,1.0);
    //FragColor = mix(texture2D(texture_diffuse1, TexCoord), texture2D(texture_height1, TexCoord), 0.2f);
      //FragColor = texture2D(texture_diffuse1, TexCoord);
}