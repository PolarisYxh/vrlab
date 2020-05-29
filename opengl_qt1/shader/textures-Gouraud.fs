#version 330 core
out vec4 FragColor;
 
in vec2 TexCoord;
 
uniform sampler2D texture_diffuse1;
uniform sampler2D texture_height1;
 
void main()
{
    FragColor = mix(texture2D(texture_diffuse1, TexCoord), texture2D(texture_height1, TexCoord), 0.2f);
      //FragColor = texture2D(texture_diffuse1, TexCoord)£»
}