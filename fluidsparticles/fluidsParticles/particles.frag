#version 120



varying float fs_pointSize;

varying vec3 fs_PosEye;
varying mat4 u_Persp;

void main(void)
{
    // calculate normal from texture coordinates
    vec3 N;

    N.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);

    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);
    
    //calculate depth
    vec4 pixelPos = vec4(fs_PosEye + normalize(N)*fs_pointSize,1.0f);
    vec4 clipSpacePos = u_Persp * pixelPos;
    //gl_FragDepth = clipSpacePos.z / clipSpacePos.w;
    
    gl_FragColor = vec4(exp(-mag*mag)*gl_Color.rgb,1.0f);
	//gl_FragColor = vec4(vec3(0.03f),1.0f);
}
