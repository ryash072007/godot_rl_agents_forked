RSRC                     VisualShader            ��������                                            w      resource_local_to_scene    resource_name    noise_type    seed 
   frequency    offset    fractal_type    fractal_octaves    fractal_lacunarity    fractal_gain    fractal_weighted_strength    fractal_ping_pong_strength    cellular_distance_function    cellular_jitter    cellular_return_type    domain_warp_enabled    domain_warp_type    domain_warp_amplitude    domain_warp_frequency    domain_warp_fractal_type    domain_warp_fractal_octaves    domain_warp_fractal_lacunarity    domain_warp_fractal_gain    script    width    height    invert    in_3d_space    generate_mipmaps 	   seamless    seamless_blend_skirt    as_normal_map    bump_strength    color_ramp    noise    output_port_for_preview    default_input_values    expanded_output_ports    source    texture    texture_type    input_name    op_type 	   operator    interpolation_mode    offsets    colors 	   function    code    graph_offset    mode    modes/blend    modes/depth_draw    modes/cull    modes/diffuse    modes/specular    flags/depth_prepass_alpha    flags/depth_test_disabled    flags/sss_mode_skin    flags/unshaded    flags/wireframe    flags/skip_vertex_transform    flags/world_vertex_coords    flags/ensure_correct_normals    flags/shadows_disabled    flags/ambient_light_disabled    flags/shadow_to_opacity    flags/vertex_lighting    flags/particle_trails    flags/alpha_to_coverage     flags/alpha_to_coverage_and_one    nodes/vertex/0/position    nodes/vertex/2/node    nodes/vertex/2/position    nodes/vertex/3/node    nodes/vertex/3/position    nodes/vertex/4/node    nodes/vertex/4/position    nodes/vertex/5/node    nodes/vertex/5/position    nodes/vertex/6/node    nodes/vertex/6/position    nodes/vertex/7/node    nodes/vertex/7/position    nodes/vertex/8/node    nodes/vertex/8/position    nodes/vertex/9/node    nodes/vertex/9/position    nodes/vertex/10/node    nodes/vertex/10/position    nodes/vertex/12/node    nodes/vertex/12/position    nodes/vertex/14/node    nodes/vertex/14/position    nodes/vertex/15/node    nodes/vertex/15/position    nodes/vertex/16/node    nodes/vertex/16/position    nodes/vertex/connections    nodes/fragment/0/position    nodes/fragment/2/node    nodes/fragment/2/position    nodes/fragment/connections    nodes/light/0/position    nodes/light/connections    nodes/start/0/position    nodes/start/connections    nodes/process/0/position    nodes/process/connections    nodes/collide/0/position    nodes/collide/connections    nodes/start_custom/0/position    nodes/start_custom/connections     nodes/process_custom/0/position !   nodes/process_custom/connections    nodes/sky/0/position    nodes/sky/connections    nodes/fog/0/position    nodes/fog/connections           local://FastNoiseLite_o7kxi          local://NoiseTexture2D_edjy2 7      &   local://VisualShaderNodeTexture_qh6jw r      $   local://VisualShaderNodeInput_s38b5 �      $   local://VisualShaderNodeInput_r4c3i �      '   local://VisualShaderNodeVectorOp_q3xvg       '   local://VisualShaderNodeVectorOp_scofx M      '   local://VisualShaderNodeVectorOp_6ruh3 �         local://Gradient_ahkpy #         local://FastNoiseLite_4w8xt �         local://NoiseTexture2D_itogf �      &   local://VisualShaderNodeTexture_n18h0 �      $   local://VisualShaderNodeInput_7urfc =      (   local://VisualShaderNodeFloatFunc_exscb t      &   local://VisualShaderNodeFloatOp_kpo51 �      (   local://VisualShaderNodeFloatFunc_3mfuq �      ,   local://VisualShaderNodeVectorCompose_dffsk 4      '   local://VisualShaderNodeVectorOp_qihxt n      $   local://VisualShaderNodeInput_v2xaq �         local://VisualShader_6h4vh          FastNoiseLite             NoiseTexture2D             �A"                      VisualShaderNodeTexture    '                     VisualShaderNodeInput    )         uv          VisualShaderNodeInput    )         vertex          VisualShaderNodeVectorOp    +                  VisualShaderNodeVectorOp    $                                        ���>���>���>+                  VisualShaderNodeVectorOp    $                     �?  �?  �?                           	   Gradient    -   !          /�>O�~?.   $        �?          �?  �?���>�B<  �?  �?�"I?f�<  �?         FastNoiseLite             NoiseTexture2D    !            "         	            VisualShaderNodeTexture    #          '         
   (                  VisualShaderNodeInput    )         time          VisualShaderNodeFloatFunc    /                  VisualShaderNodeFloatOp    $                                  �B         VisualShaderNodeFloatFunc    /                   VisualShaderNodeVectorCompose    *                   VisualShaderNodeVectorOp    $                
                 
           *                   VisualShaderNodeInput    )         color          VisualShader #   0      S  shader_type spatial;
uniform sampler2D tex_vtx_2;
uniform sampler2D tex_vtx_8 : source_color;



void vertex() {
// Input:9
	float n_out9p0 = TIME;


// FloatOp:12
	float n_in12p1 = 104.00000;
	float n_out12p0 = n_out9p0 + n_in12p1;


// FloatFunc:10
	float n_out10p0 = cos(n_out12p0);


// FloatFunc:14
	float n_out14p0 = sin(n_out12p0);


// VectorCompose:15
	vec2 n_out15p0 = vec2(n_out10p0, n_out14p0);


// Input:3
	vec2 n_out3p0 = UV;


// VectorOp:16
	vec2 n_out16p0 = n_out15p0 + n_out3p0;


// Texture2D:2
	vec4 n_out2p0 = texture(tex_vtx_2, n_out16p0);


// VectorOp:6
	vec3 n_in6p1 = vec3(0.40000, 0.40000, 0.40000);
	vec3 n_out6p0 = vec3(n_out2p0.xyz) * n_in6p1;


// VectorOp:7
	vec3 n_in7p0 = vec3(1.00000, 1.00000, 1.00000);
	vec3 n_out7p0 = n_in7p0 + n_out6p0;


// Input:4
	vec3 n_out4p0 = VERTEX;


// VectorOp:5
	vec3 n_out5p0 = n_out7p0 * n_out4p0;


// Texture2D:8
	vec4 n_out8p0 = texture(tex_vtx_8, n_out3p0);


// Output:0
	VERTEX = n_out5p0;
	COLOR.rgb = vec3(n_out8p0.xyz);


}

void fragment() {
// Input:2
	vec4 n_out2p0 = COLOR;


// Output:0
	ALBEDO = vec3(n_out2p0.xyz);


}
 1   
   �ED/=��G   
    ��D  �BH            I   
     RD  *�J            K   
      C  �L            M   
     �D  ��N            O   
     �D  �P            Q   
     D  �R            S   
    ��D  �T            U   
     aD  H�V            W   
     ��  z�X            Y   
     �B  u�Z            [   
     ��  z�\            ]   
     pB  W�^            _   
     �C  H�`            a   
     D  *�b       <                                                                                       	                                   
       
                                                                                 c   
    ��D  p�d            e   
   �ӐD��;�f                               RSRC