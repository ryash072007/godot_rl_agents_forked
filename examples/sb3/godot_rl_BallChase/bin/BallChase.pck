GDPC                                                                                         &   d   res://.godot/exported/133200997/export-1222b1ceeacbc57c3f8fe0e4235d9be0-ExampleRaycastSensor3D.scn  @      �      xX���u��b�x    \   res://.godot/exported/133200997/export-1cc1ad7ce1f98ed0c4cee4b060bc26fd-RaycastSensor3D.scn �#      �      �K��EB�0�NR1W�l-    d   res://.godot/exported/133200997/export-5127fc3fc2907f1e87ac8e869558593d-ExampleRaycastSensor2D.scn          H      )O\Ih�4�M�u��I�    \   res://.godot/exported/133200997/export-7454c66d21916090bf5dc7766fa8629a-RaycastSensor2D.scn �      �      1(W}8�O��V ����    X   res://.godot/exported/133200997/export-7cf3fd67ad9f55210191d77b582b8209-default_env.res ��      �	      ��(�aA��llG���}�    `   res://.godot/exported/133200997/export-d1d9cc1906d18d9a32f853b74a4d90f4-RGBCameraSensor3D.scn   �)      �      ��u�]�:�륡"$    X   res://.godot/exported/133200997/export-e0c93c3e5af95d0922ba36d1569eeaa1-BatchEnvs.scn   �l      s      "ܑ�6�kT���Vݻ    X   res://.godot/exported/133200997/export-f42ac285715e233e30405f3a6a1edaf9-BallChase.scn   �S      �      �����B��&Nh��3    H   res://.godot/imported/cherry.png-54f14847dd7a94d627024962d7d6b865.ctex   x      �      �K�,��|��ԥ�    D   res://.godot/imported/icon.png-45a871b53434e556222f5901d598ab34.ctex�2      �       ��H�g~ @"�:SU�    D   res://.godot/imported/icon.png-487276ed1e3a0c39cad0279d744ee560.ctex��      �      �`� ͏+�>��`    L   res://.godot/imported/light_mask.png-40da3c93e1795f65c34ad69a6ae38ba3.ctex  ��      F       ��u��)hx�YC�P��    P   res://.godot/imported/lollipopGreen.png-ad04020a819959359f29965dcb47712e.ctex   ж            ����t�BmF����9�       res://.godot/uid_cache.bin  0�      E      T�[>gN-�^C� �       res://BallChase.tscn.remap  ��      f       >[��iq����5�t       res://BatchEnvs.tscn.remap  `�      f       �nf�)=?1���M1d6       res://Camera2D.gd   0t      �      	���7�m��%š�        res://Player.gd �      �      �6����st��˱&1    0   res://addons/godot_rl_agents/godot_rl_agents.gd `0            }f�<i�S�.��t��4    ,   res://addons/godot_rl_agents/icon.png.importp3      �      쓀��+8��\xP\    T   res://addons/godot_rl_agents/sensors/sensors_2d/ExampleRaycastSensor2D.tscn.remap   ��      s       �U����OA����W    <   res://addons/godot_rl_agents/sensors/sensors_2d/ISensor2D.gdP      �       ��=ohk���*��    D   res://addons/godot_rl_agents/sensors/sensors_2d/RaycastSensor2D.gd  @      F      S_�N����:���    L   res://addons/godot_rl_agents/sensors/sensors_2d/RaycastSensor2D.tscn.remap   �      l       {�ڹ��w3V��q^    T   res://addons/godot_rl_agents/sensors/sensors_3d/ExampleRaycastSensor3D.tscn.remap   ��      s       )|�*�5�'�W��Q    <   res://addons/godot_rl_agents/sensors/sensors_3d/ISensor3D.gd�      �       ��Y�F��R����b    D   res://addons/godot_rl_agents/sensors/sensors_3d/RGBCameraSensor3D.gdp(      [      ��EGiꇩ��2V�4    L   res://addons/godot_rl_agents/sensors/sensors_3d/RGBCameraSensor3D.tscn.remap��      n       ��C���hP�2��    D   res://addons/godot_rl_agents/sensors/sensors_3d/RaycastSensor3D.gd  �      �      �]L�!'/V2<�b�%    L   res://addons/godot_rl_agents/sensors/sensors_3d/RaycastSensor3D.tscn.remap  �      l       �Rُ�2��s�;�ok    $   res://addons/godot_rl_agents/sync.gdp6      E      Z�r�5�u��xSi���       res://cherry.png.import  ~      �      �=x������XJ��ɜ       res://default_env.tres.remap��      h       cXv�S��P�O�Tq�o       res://icon.png  @�      �      G1?��z�c��vN��       res://icon.png.import   ��      �      mn�{�A����䅿��       res://light_mask.png.import г      �      ��G�!5v�O��c��        res://lollipopGreen.png.import  �      �      �R������@����|       res://project.binary��      1      (n�R�C��'S�N�Y�    ��>�RSRC                     PackedScene            ��������                                                  resource_local_to_scene    resource_name    script/source    custom_solver_bias    radius    script 	   _bundled       Script ,   res://sensors/sensors_2d/RaycastSensor2D.gd ��������      local://GDScript_jbxey �         local://GDScript_dargt ?         local://CircleShape2D_5fkp7 '         local://PackedScene_s2mby E      	   GDScript          ]   extends Node2D



func _physics_process(delta: float) -> void:
    print("step start")
    
 	   GDScript          �   extends RayCast2D

var steps = 1

func _physics_process(delta: float) -> void:
    print("processing raycast")
    steps += 1
    if steps % 2:
        force_raycast_update()

    print(is_colliding())
    CircleShape2D             PackedScene          	         names "         ExampleRaycastSensor2D    script    Node2D    ExampleAgent 	   position 	   rotation    RaycastSensor2D    TestRayCast2D 
   RayCast2D    StaticBody2D    CollisionShape2D    shape    	   variants                 
    @D  �C   C��>                   
     �?  PB               node_count             nodes     8   ��������       ����                            ����                                ����                           ����                     	   	   ����                    
   
   ����                   conn_count              conns               node_paths              editable_instances              version             RSRC~�C>&�Hextends Node2D
class_name ISensor2D

var _obs : Array
var _active := false

func get_observation():
	pass
	
func activate():
	_active = true
	
func deactivate():
	_active = false

func _update_observation():
	pass
	
func reset():
	pass
c�(Bextends ISensor2D
class_name RaycastSensor2D
@tool

@export var n_rays := 16.0:
	get: return n_rays
	set(value):
		n_rays = value
		_update()
	
@export var ray_length := 200:# (float,5,200,5.0)
	get: return ray_length
	set(value):
		ray_length = value
		_update()
@export var cone_width := 360.0:# (float,5,360,5.0)
	get: return cone_width
	set(value):
		cone_width = value
		_update()
	
@export var debug_draw := false :
	get: return debug_draw 
	set(value):
		debug_draw = value
		_update()  


var _angles = []
var rays := []

func _update():
	if Engine.is_editor_hint():
		_spawn_nodes()	

func _ready() -> void:
	_spawn_nodes()


func _spawn_nodes():
	for ray in rays:
		ray.queue_free()
	rays = []
		
	_angles = []
	var step = cone_width / (n_rays)
	var start = step/2 - cone_width/2
	
	for i in n_rays:
		var angle = start + i * step
		var ray = RayCast2D.new()
		ray.set_target_position(Vector2(
			ray_length*cos(deg_to_rad(angle)),
			ray_length*sin(deg_to_rad(angle))
		))
		ray.set_name("node_"+str(i))
		ray.enabled  = true
		ray.collide_with_areas = true
		add_child(ray)
		rays.append(ray)
		
		
		_angles.append(start + i * step)
	

func _physics_process(delta: float) -> void:
	if self._active:
		self._obs = calculate_raycasts()
		
func get_observation() -> Array:
	if len(self._obs) == 0:
		print("obs was null, forcing raycast update")
		return self.calculate_raycasts()
	return self._obs
	

func calculate_raycasts() -> Array:
	var result = []
	for ray in rays:
		ray.force_raycast_update()
		var distance = _get_raycast_distance(ray)
		result.append(distance)
	return result

func _get_raycast_distance(ray : RayCast2D) -> float : 
	if !ray.is_colliding():
		return 0.0
		
	var distance = (global_position - ray.get_collision_point()).length()
	distance = clamp(distance, 0.0, ray_length)
	return (ray_length - distance) / ray_length
	
	
	
g�������RSRC                     PackedScene            ��������                                                  resource_local_to_scene    resource_name 	   _bundled    script       Script C   res://addons/godot_rl_agents/sensors/sensors_2d/RaycastSensor2D.gd ��������      local://PackedScene_v7uvu :         PackedScene          	         names "         RaycastSensor2D    script    Node2D    	   variants                       node_count             nodes     	   ��������       ����                    conn_count              conns               node_paths              editable_instances              version             RSRC����cG�4jRSRC                     PackedScene            ��������                                                  resource_local_to_scene    resource_name 	   _bundled    script           local://PackedScene_cusqc �          PackedScene          	         names "         ExampleRaycastSensor3D    Node3D 	   Camera3D 
   transform    	   variants            �?              �?              �?��M?    ��,@      node_count             nodes        ��������       ����                      ����                    conn_count              conns               node_paths              editable_instances              version             RSRC��Vm��extends Node3D
class_name ISensor3D

var _obs : Array
var _active := false

func get_observation():
	pass
	
func activate():
	_active = true
	
func deactivate():
	_active = false

func _update_observation():
	pass
	
func reset():
	pass
�}��extends ISensor3D
class_name RayCastSensor3D
@tool
@export_flags_3d_physics var collision_mask = 1:
	get: return collision_mask
	set(value):
		collision_mask = value
		_update()
@export_flags_3d_physics var boolean_class_mask = 1:
	get: return boolean_class_mask
	set(value):
		boolean_class_mask = value
		_update()		

@export var n_rays_width := 6.0:
	get: return n_rays_width
	set(value):
		n_rays_width = value
		_update()
	
@export var n_rays_height := 6.0:
	get: return n_rays_height
	set(value):
		n_rays_height = value
		_update()

@export var ray_length := 10.0:
	get: return ray_length
	set(value):
		ray_length = value
		_update()
		
@export var cone_width := 60.0:
	get: return cone_width
	set(value):
		cone_width = value
		_update()
		
@export var cone_height := 60.0:
	get: return cone_height
	set(value):
		cone_height = value
		_update()

@export var collide_with_bodies := true:
	get: return collide_with_bodies
	set(value):
		collide_with_bodies = value
		_update()
		
@export var collide_with_areas := false:
	get: return collide_with_areas
	set(value):
		collide_with_areas = value
		_update()
		
@export var class_sensor := false
		
var rays := []
var geo = null

func _update():
	if Engine.is_editor_hint():
		_spawn_nodes()	


func _ready() -> void:
	_spawn_nodes()

func _spawn_nodes():
	print("spawning nodes")
	for ray in get_children():
		ray.queue_free()
	if geo:
		geo.clear()
	#$Lines.remove_points()
	rays = []
	
	var horizontal_step = cone_width / (n_rays_width)
	var vertical_step = cone_height / (n_rays_height)
	
	var horizontal_start = horizontal_step/2 - cone_width/2
	var vertical_start = vertical_step/2 - cone_height/2   

	var points = []
	
	for i in n_rays_width:
		for j in n_rays_height:
			var angle_w = horizontal_start + i * horizontal_step
			var angle_h = vertical_start + j * vertical_step
			#angle_h = 0.0
			var ray = RayCast3D.new()
			var cast_to = to_spherical_coords(ray_length, angle_w, angle_h)
			ray.set_target_position(cast_to)

			points.append(cast_to)
			
			ray.set_name("node_"+str(i)+" "+str(j))
			ray.enabled  = true
			ray.collide_with_bodies = collide_with_bodies
			ray.collide_with_areas = collide_with_areas
			ray.collision_mask = collision_mask
			add_child(ray)
			ray.set_owner(get_tree().edited_scene_root)
			rays.append(ray)
			ray.force_raycast_update()
			
#    if Engine.editor_hint:
#        _create_debug_lines(points)
		
func _create_debug_lines(points):
	if not geo: 
		geo = ImmediateMesh.new()
		add_child(geo)
		
	geo.clear()
	geo.begin(Mesh.PRIMITIVE_LINES)
	for point in points:
		geo.set_color(Color.AQUA)
		geo.add_vertex(Vector3.ZERO)
		geo.add_vertex(point)
	geo.end()

func display():
	if geo:
		geo.display()
	
func to_spherical_coords(r, inc, azimuth) -> Vector3:
	return Vector3(
		r*sin(deg_to_rad(inc))*cos(deg_to_rad(azimuth)),
		r*sin(deg_to_rad(azimuth)),
		r*cos(deg_to_rad(inc))*cos(deg_to_rad(azimuth))       
	)
	
func get_observation() -> Array:
	return self.calculate_raycasts()

func calculate_raycasts() -> Array:
	var result = []
	for ray in rays:
		ray.set_enabled(true)
		ray.force_raycast_update()
		var distance = _get_raycast_distance(ray)

		result.append(distance)
		if class_sensor:
			var hit_class = 0 
			if ray.get_collider():
				var hit_collision_layer = ray.get_collider().collision_layer
				hit_collision_layer = hit_collision_layer & collision_mask
				hit_class = (hit_collision_layer & boolean_class_mask) > 0
			result.append(hit_class)
		ray.set_enabled(false)
	return result

func _get_raycast_distance(ray : RayCast3D) -> float : 
	if !ray.is_colliding():
		return 0.0
		
	var distance = (global_transform.origin - ray.get_collision_point()).length()
	distance = clamp(distance, 0.0, ray_length)
	return (ray_length - distance) / ray_length
Ϗ�5��8�$
RSRC                     PackedScene            ��������                                                  resource_local_to_scene    resource_name 	   _bundled    script       Script C   res://addons/godot_rl_agents/sensors/sensors_3d/RaycastSensor3D.gd ��������      local://PackedScene_j0gkn :         PackedScene          	         names "         RaycastSensor3D    script    n_rays_width    n_rays_height    ray_length    Node3D 	   node_0 2    collide_with_areas 
   RayCast3D 	   node_0 3 	   node_1 0 	   node_1 1 	   node_2 0 	   node_2 1 	   node_3 0 	   node_3 1    	   variants                      �@      @     0A            node_count    	         nodes     W   ��������       ����                                              ����                        	   ����                        
   ����                           ����                           ����                           ����                           ����                           ����                   conn_count              conns               node_paths              editable_instances              version             RSRCN;extends Node3D
class_name RGBCameraSensor3D
var camera_pixels = null

@onready var camera_texture := $Control/TextureRect/CameraTexture as Sprite2D

func get_camera_pixel_encoding():
	return camera_texture.get_texture().get_data().data["data"].hex_encode()

func get_camera_shape()-> Array:
	return [$SubViewport.size[0], $SubViewport.size[1], 4]
h	�7RSRC                     PackedScene            ��������                                                  ..    SubViewport 	   Camera3D    resource_local_to_scene    resource_name    viewport_path    script 	   _bundled       Script E   res://addons/godot_rl_agents/sensors/sensors_3d/RGBCameraSensor3D.gd ��������      local://ViewportTexture_hww16 �         local://PackedScene_tpopq �         ViewportTexture                         PackedScene          	         names "         RGBCameraSensor3D    script    Node3D    RemoteTransform3D    remote_path    SubViewport    size    render_target_update_mode 	   Camera3D    near    Control    layout_mode    anchors_preset    anchor_right    anchor_bottom    TextureRect    offset_left    offset_top    offset_right    offset_bottom    scale    color 
   ColorRect    CameraTexture    texture    offset    flip_v 	   Sprite2D    	   variants                                 -                       ?           �?     �D    �D    @�D     
D
      A   A   �� <�� <�� <  �?          
     A  A            node_count             nodes     U   ��������       ����                            ����                           ����                                ����   	                  
   
   ����                                            ����                  	      
                                ����                               conn_count              conns               node_paths              editable_instances              version             RSRC<��KCͧxK�@tool
extends EditorPlugin


func _enter_tree():
	# Initialization of the plugin goes here.
	# Add the new type with a name, a parent type, a script and an icon.
	add_custom_type("Sync", "Node", preload("sync.gd"), preload("icon.png"))
	#add_custom_type("RaycastSensor2D2", "Node", preload("raycast_sensor_2d.gd"), preload("icon.png"))


func _exit_tree():
	# Clean-up of the plugin goes here.
	# Always remember to remove it from the engine when deactivated.
	remove_custom_type("Sync")
	#remove_custom_type("RaycastSensor2D2")
4"T�bE��݈Y�GST2            ����                        �   RIFF�   WEBPVP8L�   /��`�m�~��~�6�?�1g m�����cE���\�Q@ �+����SQ��88�a���;[�w�
#�m�tPV����"�?X	�;`&�� x�uHǅ��Fi�^��(�V��<ǲbkf���X�pM�4��w����J��^���U��M]R 2�\�[[remap]

importer="texture"
type="CompressedTexture2D"
uid="uid://b7lwg3uk8v3pr"
path="res://.godot/imported/icon.png-45a871b53434e556222f5901d598ab34.ctex"
metadata={
"vram_texture": false
}

[deps]

source_file="res://addons/godot_rl_agents/icon.png"
dest_files=["res://.godot/imported/icon.png-45a871b53434e556222f5901d598ab34.ctex"]

[params]

compress/mode=0
compress/lossy_quality=0.7
compress/hdr_compression=1
compress/bptc_ldr=0
compress/normal_map=0
compress/channel_pack=0
mipmaps/generate=false
mipmaps/limit=-1
roughness/mode=0
roughness/src_normal=""
process/fix_alpha_border=true
process/premult_alpha=false
process/normal_map_invert_y=false
process/hdr_as_srgb=false
process/hdr_clamp_exposure=false
process/size_limit=0
detect_3d/compress_to=1
a��]�QNextends Node
# --fixed-fps 2000 --disable-render-loop
@export var action_repeat := 8
@export var speed_up = 1
var n_action_steps = 0

const MAJOR_VERSION := "0"
const MINOR_VERSION := "3" 
const DEFAULT_PORT := "11008"
const DEFAULT_SEED := "1"
const DEFAULT_ACTION_REPEAT := "8"
var stream : StreamPeerTCP = null
var connected = false
var message_center
var should_connect = true
var agents
var need_to_send_obs = false
var args = null
@onready var start_time = Time.get_ticks_msec()
var initialized = false
var just_reset = false


# Called when the node enters the scene tree for the first time.

func _ready():

	await get_tree().root.ready
	get_tree().set_pause(true) 
	_initialize()
	await get_tree().create_timer(1.0).timeout
	get_tree().set_pause(false) 
		
func _get_agents():
	agents = get_tree().get_nodes_in_group("AGENT")

func _set_heuristic(heuristic):
	for agent in agents:
		agent.set_heuristic(heuristic)

func _handshake():
	print("performing handshake")
	
	var json_dict = _get_dict_json_message()
	assert(json_dict["type"] == "handshake")
	var major_version = json_dict["major_version"]
	var minor_version = json_dict["minor_version"]
	if major_version != MAJOR_VERSION:
		print("WARNING: major verison mismatch ", major_version, " ", MAJOR_VERSION)  
	if minor_version != MINOR_VERSION:
		print("WARNING: minor verison mismatch ", minor_version, " ", MINOR_VERSION)
		
	print("handshake complete")

func _get_dict_json_message():
	# returns a dictionary from of the most recent message
	# this is not waiting
	while stream.get_available_bytes() == 0:
		stream.poll()
		if stream.get_status() != 2:
			print("server disconnected status, closing")
			get_tree().quit()
			return null

		OS.delay_usec(10)
		
	var message = stream.get_string()
	var json_data = JSON.parse_string(message)
	
	return json_data

func _send_dict_as_json_message(dict):
	stream.put_string(JSON.stringify(dict))

func _send_env_info():
	var json_dict = _get_dict_json_message()
	assert(json_dict["type"] == "env_info")
	
	var message = {
		"type" : "env_info",
		#"obs_size": agents[0].get_obs_size(),
		"observation_space": agents[0].get_obs_space(),
		"action_space":agents[0].get_action_space(),
		"n_agents": len(agents)
		}
	_send_dict_as_json_message(message)


func connect_to_server():
	print("Waiting for one second to allow server to start")
	OS.delay_msec(1000)
	print("trying to connect to server")
	stream = StreamPeerTCP.new()
	
	# "localhost" was not working on windows VM, had to use the IP
	var ip = "127.0.0.1"
	var port = _get_port()
	var connect = stream.connect_to_host(ip, port)
	stream.set_no_delay(true) # TODO check if this improves performance or not
	stream.poll()
	return stream.get_status() == 2

func _get_args():
	print("getting command line arguments")
#	var arguments = {}
#	for argument in OS.get_cmdline_args():
#		# Parse valid command-line arguments into a dictionary
#		if argument.find("=") > -1:
#			var key_value = argument.split("=")
#			arguments[key_value[0].lstrip("--")] = key_value[1]
			
	var arguments = {}
	for argument in OS.get_cmdline_args():
		print(argument)
		if argument.find("=") > -1:
			var key_value = argument.split("=")
			arguments[key_value[0].lstrip("--")] = key_value[1]
		else:
			# Options without an argument will be present in the dictionary,
			# with the value set to an empty string.
			arguments[argument.lstrip("--")] = ""

	return arguments   

func _get_speedup():
	print(args)
	return args.get("speedup", str(speed_up)).to_int()

func _get_port():    
	return args.get("port", DEFAULT_PORT).to_int()

func _set_seed():
	var _seed = args.get("env_seed", DEFAULT_SEED).to_int()
	seed(_seed)

func _set_action_repeat():
	action_repeat = args.get("action_repeat", DEFAULT_ACTION_REPEAT).to_int()
	
func disconnect_from_server():
	stream.disconnect_from_host()

func _initialize():
	_get_agents()
	
	args = _get_args()
	Engine.physics_ticks_per_second = _get_speedup() * 60 # Replace with function body.
	Engine.time_scale = _get_speedup() * 1.0
	prints("physics ticks", Engine.physics_ticks_per_second, Engine.time_scale, _get_speedup(), speed_up)
	
	connected = connect_to_server()
	if connected:
		_set_heuristic("model")
		_handshake()
		_send_env_info()
	else:
		_set_heuristic("human")  
		
	_set_seed()
	_set_action_repeat()
	initialized = true  

func _physics_process(delta): 
	# two modes, human control, agent control
	# pause tree, send obs, get actions, set actions, unpause tree
	if n_action_steps % action_repeat != 0:
		n_action_steps += 1
		return

	n_action_steps += 1
	
	if connected:
		get_tree().set_pause(true) 
		
		if just_reset:
			just_reset = false
			var obs = _get_obs_from_agents()
		
			var reply = {
				"type": "reset",
				"obs": obs
			}
			_send_dict_as_json_message(reply)
			# this should go straight to getting the action and setting it checked the agent, no need to perform one phyics tick
			get_tree().set_pause(false) 
			return
		
		if need_to_send_obs:
			need_to_send_obs = false
			var reward = _get_reward_from_agents()
			var done = _get_done_from_agents()
			#_reset_agents_if_done() # this ensures the new observation is from the next env instance : NEEDS REFACTOR
			
			var obs = _get_obs_from_agents()
			
			var reply = {
				"type": "step",
				"obs": obs,
				"reward": reward,
				"done": done
			}
			_send_dict_as_json_message(reply)
		
		var handled = handle_message()
	else:
		_reset_agents_if_done()

func handle_message() -> bool:
	# get json message: reset, step, close
	var message = _get_dict_json_message()
	if message["type"] == "close":
		print("received close message, closing game")
		get_tree().quit()
		get_tree().set_pause(false) 
		return true
		
	if message["type"] == "reset":
		print("resetting all agents")
		_reset_all_agents()
		just_reset = true
		get_tree().set_pause(false) 
		#print("resetting forcing draw")
#        RenderingServer.force_draw()
#        var obs = _get_obs_from_agents()
#        print("obs ", obs)
#        var reply = {
#            "type": "reset",
#            "obs": obs
#        }
#        _send_dict_as_json_message(reply)   
		return true
		
	if message["type"] == "call":
		var method = message["method"]
		var returns = _call_method_on_agents(method)
		var reply = {
			"type": "call",
			"returns": returns
		}
		print("calling method from Python")
		_send_dict_as_json_message(reply)   
		return handle_message()
	
	if message["type"] == "action":
		var action = message["action"]
		_set_agent_actions(action) 
		need_to_send_obs = true
		get_tree().set_pause(false) 
		return true
		
	print("message was not handled")
	return false

func _call_method_on_agents(method):
	var returns = []
	for agent in agents:
		returns.append(agent.call(method))
		
	return returns


func _reset_agents_if_done():
	for agent in agents:
		if agent.get_done(): 
			agent.set_done_false()

func _reset_all_agents():
	for agent in agents:
		agent.needs_reset = true
		#agent.reset()   

func _get_obs_from_agents():
	var obs = []
	for agent in agents:
		obs.append(agent.get_obs())
	return obs
	
func _get_reward_from_agents():
	var rewards = [] 
	for agent in agents:
		rewards.append(agent.get_reward())
		agent.zero_reward()
	return rewards    
	
func _get_done_from_agents():
	var dones = [] 
	for agent in agents:
		var done = agent.get_done()
		if done: agent.set_done_false()
		dones.append(done)
	return dones    
	
func _set_agent_actions(actions):
	for i in range(len(actions)):
		agents[i].set_action(actions[i])
	
<��T��Cuq�RSRC                     PackedScene            ��������                                            e      resource_local_to_scene    resource_name    custom_solver_bias    radius    script    size    background_mode    background_color    background_energy_multiplier    background_intensity    background_canvas_max_layer    background_camera_feed_id    sky    sky_custom_fov    sky_rotation    ambient_light_source    ambient_light_color    ambient_light_sky_contribution    ambient_light_energy    reflected_light_source    tonemap_mode    tonemap_exposure    tonemap_white    ssr_enabled    ssr_max_steps    ssr_fade_in    ssr_fade_out    ssr_depth_tolerance    ssao_enabled    ssao_radius    ssao_intensity    ssao_power    ssao_detail    ssao_horizon    ssao_sharpness    ssao_light_affect    ssao_ao_channel_affect    ssil_enabled    ssil_radius    ssil_intensity    ssil_sharpness    ssil_normal_rejection    sdfgi_enabled    sdfgi_use_occlusion    sdfgi_read_sky_light    sdfgi_bounce_feedback    sdfgi_cascades    sdfgi_min_cell_size    sdfgi_cascade0_distance    sdfgi_max_distance    sdfgi_y_scale    sdfgi_energy    sdfgi_normal_bias    sdfgi_probe_bias    glow_enabled    glow_levels/1    glow_levels/2    glow_levels/3    glow_levels/4    glow_levels/5    glow_levels/6    glow_levels/7    glow_normalized    glow_intensity    glow_strength 	   glow_mix    glow_bloom    glow_blend_mode    glow_hdr_threshold    glow_hdr_scale    glow_hdr_luminance_cap    glow_map_strength 	   glow_map    fog_enabled    fog_light_color    fog_light_energy    fog_sun_scatter    fog_density    fog_aerial_perspective    fog_sky_affect    fog_height    fog_height_density    volumetric_fog_enabled    volumetric_fog_density    volumetric_fog_albedo    volumetric_fog_emission    volumetric_fog_emission_energy    volumetric_fog_gi_inject    volumetric_fog_anisotropy    volumetric_fog_length    volumetric_fog_detail_spread    volumetric_fog_ambient_inject    volumetric_fog_sky_affect -   volumetric_fog_temporal_reprojection_enabled ,   volumetric_fog_temporal_reprojection_amount    adjustment_enabled    adjustment_brightness    adjustment_contrast    adjustment_saturation    adjustment_color_correction 	   _bundled       Script    res://Player.gd ��������   Script C   res://addons/godot_rl_agents/sensors/sensors_2d/RaycastSensor2D.gd ��������
   Texture2D    res://lollipopGreen.png ���qZ
   Texture2D    res://cherry.png )�E4WnUb	      local://CircleShape2D_om33k j         local://CircleShape2D_qvv7u �         local://RectangleShape2D_vgaj1 �         local://RectangleShape2D_oehmj �         local://RectangleShape2D_5alxn           local://RectangleShape2D_xlbbn Q         local://RectangleShape2D_qx8n6 �         local://Environment_2rwpj �         local://PackedScene_i25ic S         CircleShape2D            �A         CircleShape2D            �A         RectangleShape2D       
     �A  4D         RectangleShape2D       
     �A  4D         RectangleShape2D       
     �D  �A         RectangleShape2D       
     �D  �A         RectangleShape2D       
   ZiB+'�C         Environment                      7        �?8        �?9          :        �?;          ?      ���=@      �?B      �Q�=C                   PackedScene    d      	         names "   ,   
   BallChase    Node2D    Player 	   position    script    AGENT    CharacterBody2D    CollisionShape2D    shape    RaycastSensor2D 	   Sprite2D    scale    texture    Fruit    collision_layer    collision_mask    Area2D    Walls 	   LeftWall 
   ColorRect    offset_left    offset_top    offset_right    offset_bottom    color 
   RightWall    TopWall    BottomWall 
   Obstacle4 
   Obstacle5    BackGround    layer    CanvasLayer    anchors_preset    anchor_right    anchor_bottom    WorldEnvironment    environment    _on_Fruit_body_entered    body_entered    _on_LeftWall_body_entered    _on_RightWall_body_entered    _on_TopWall_body_entered    _on_BottomWall_body_entered    	   variants    +   
   /݌Ch�"C                             
   fff?fff?         
   R��DV�D                     
   �[@��c�
   ���?���?         
      A  �C               �     ��      A     �C   ���>��?��?  �?
    ��D  �C         
      D   A              �     D
      D �1D         
     `D  �C              �     �     �A     C
     �C  �C   ����   ����   +��@   �Q�@   �KrF   f��E   ��L>��0>���>  �?               node_count             nodes     Q  ��������       ����                      ����                                   ����                       	   ����                    
   
   ����                                 ����                                      ����      	              
   
   ����      
                                 ����                     ����             	             ����             	             ����                                                  ����                          ����                          ����                                                  ����                          ����                          ����                                                  ����                          ����                          ����                                                  ����                          ����                          ����                         !                          ����      "                    ����                          ����                         !                            ����      #                    ����   !   $   "   %   #   &      '      (      )               $   $   ����   %   *             conn_count             conns     *         '   &          	      '   (                '   )                '   *                '   +                '   +                '   +                node_paths              editable_instances              version             RSRCN�c��cRSRC                     PackedScene            ��������                                                  resource_local_to_scene    resource_name 	   _bundled    script       PackedScene    res://BallChase.tscn �D!R��*   Script    res://Camera2D.gd ��������   Script %   res://addons/godot_rl_agents/sync.gd ��������      local://PackedScene_bwu7l v         PackedScene          	         names "      
   BatchEnvs    Node 
   BallChase    BallChase2 	   position    BallChase3    BallChase4    BallChase5    BallChase6    BallChase7    BallChase8    BallChase9    BallChase10    BallChase11    BallChase12    BallChase13    BallChase14    BallChase15    BallChase16 	   Camera2D    offset    current    script    Sync    	   variants                 
   �(�D    
   �(#E    
       ��<D
   �(�D��<D
   �y#E��<D
   �`��
g�D
   �!�D),�D
    �#E\��D
   ���?��E
   =��Df
E
   ��#E�pE
   {�tE    
   \+uE��;D
   qMuE�5�D
   �-uE�0E
      D  �C                              node_count             nodes     �   ��������       ����                ���                       ���                            ���                            ���                            ���                            ���                            ���	                            ���
                            ���                            ���             	               ���             
               ���                            ���                            ���                            ���                            ���                                  ����                                       ����                   conn_count              conns               node_paths              editable_instances              version             RSRC�:Dg�<��$;��extends Camera2D

const MOVE_SPEED = 1000
const MIN_ZOOM = 0.1
const MAX_ZOOM = 1
const ZOOM_FACTOR = 1.2

func _process(delta):
	if Input.is_action_pressed("reset_camera"):
		global_position = Vector2.ZERO
	if Input.is_action_pressed("left_arrow"):
		global_position += Vector2.LEFT * delta * MOVE_SPEED
	elif Input.is_action_pressed("right_arrow"):
		global_position += Vector2.RIGHT * delta * MOVE_SPEED    
	if Input.is_action_pressed("up_arrow"):
		global_position += Vector2.UP * delta * MOVE_SPEED
	elif Input.is_action_pressed("down_arrow"):
		global_position += Vector2.DOWN * delta * MOVE_SPEED
		
	global_position.x = max(0, global_position.x)
	global_position.y = max(0, global_position.y)


func _input(event : InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.is_action_pressed("zoom_in"):
			print("zoom_in")
		if event.is_action_pressed("zoom_out"):
			zoom /= ZOOM_FACTOR
	zoom.x = clamp(zoom.x, MIN_ZOOM, MAX_ZOOM)
	zoom.y = clamp(zoom.y, MIN_ZOOM, MAX_ZOOM)
�Ɛˋ�*��qGST2   F   F      ����               F F        �  RIFF�  WEBPVP8L�  /E@�1�l�J���	��;��H�#IR$y��s����$9𵼝��Hr$1�{���8����V3�im�܏ 	R��(A�P���+A�|�4�v�u���z���AZ�ڈb �d�H""�$�X��{B-���DК�^�����{4���|��˴��y���G}PU	�eլ�P�s��7�s� jX3��c�,3˰�,��%L�0�T`Du��h���+M��^�k���n&����{��>�u�W�|���?����7��4/�SlAB $$d�� P��KE��SQ!m��Y��E���ܘPs���궴U����jk��Zk[TZ�ꍋ�+��钵N��c9/��g�T�CG�>��ʌ}�+�Q��)EaKI�VE4C��i��&�FH�j���y�;>W�^���|��w��	�������m�Y۶mc�$ck�k۶�c�d�Q:k�����E�]��n9�������z�:FZw��vX`����0z)��e�~��0���=�z�ÐTG�=G�ۼm�^l}J��^�>�۲}�q�t���^�t���認[v�:�*Tϝ9�n�;�7=���cPc
}����o��R��99jll��&�ݏ7������k5���+ׯ��7��G�=����'M1^Mu�m�����X����߰��<H@�%�o����{5�<G7���5?��w;��I�:*+�CG�ڴ|ɯ|Iѓt��<Jf��%K��Ο(�W	ٰ���=k���ɐ���I.[A��ף�c�tj���\�_T
*�mؾ�/_5>N:�IeOzY��,��]���^54<a������x��c��.*j��ن�_^�y%ށm?I:���&��\cVg�Y�F�P�+z��Ć��sǫ����l�����x���M��bcVU�.�S{)�g���(�Ę��ڴ���3�ⓒj�-�.��+5��X�N��mTܐ/��1����4M˲�n����HN�L�Љ���Ç��2k��[t�J�6Q{���`6��:�}1�%���	�~;�k+]Ѕ҉Y{�<�(��z�:�#���?���S���ˌD`�����Q6[e_ϼ�{8�(1a��/�(��y�|�4�Ը8'�g�Ҩ�JLƸx��5�X`+С
��4Z�B�wi����B�	�ߩ���*���&��:0����� ���g��
*%W�u������%�>� Un����5�C(u����B=��[.���P_�����ZI�'�	z�4` $�����@B�$O�֭Ÿ�~���$q���N8@�_���Z�pf ����4�)����z� E�-M��8-�����ϡ^�`�"ʅ����ixq
Z����n�5�ℂ����ML�4��⌂�[�����^<�S
��P(�pw�c���Bs��QD�  ��H�9A�wu7O�[remap]

importer="texture"
type="CompressedTexture2D"
uid="uid://c77mj86i50gub"
path="res://.godot/imported/cherry.png-54f14847dd7a94d627024962d7d6b865.ctex"
metadata={
"vram_texture": false
}

[deps]

source_file="res://cherry.png"
dest_files=["res://.godot/imported/cherry.png-54f14847dd7a94d627024962d7d6b865.ctex"]

[params]

compress/mode=0
compress/lossy_quality=0.7
compress/hdr_compression=1
compress/bptc_ldr=0
compress/normal_map=0
compress/channel_pack=0
mipmaps/generate=false
mipmaps/limit=-1
roughness/mode=0
roughness/src_normal=""
process/fix_alpha_border=true
process/premult_alpha=false
process/normal_map_invert_y=false
process/hdr_as_srgb=false
process/hdr_clamp_exposure=false
process/size_limit=0
detect_3d/compress_to=1
��^�VRSRC                     Environment            ��������                                            d      resource_local_to_scene    resource_name    sky_material    process_mode    radiance_size    script    background_mode    background_color    background_energy_multiplier    background_intensity    background_canvas_max_layer    background_camera_feed_id    sky    sky_custom_fov    sky_rotation    ambient_light_source    ambient_light_color    ambient_light_sky_contribution    ambient_light_energy    reflected_light_source    tonemap_mode    tonemap_exposure    tonemap_white    ssr_enabled    ssr_max_steps    ssr_fade_in    ssr_fade_out    ssr_depth_tolerance    ssao_enabled    ssao_radius    ssao_intensity    ssao_power    ssao_detail    ssao_horizon    ssao_sharpness    ssao_light_affect    ssao_ao_channel_affect    ssil_enabled    ssil_radius    ssil_intensity    ssil_sharpness    ssil_normal_rejection    sdfgi_enabled    sdfgi_use_occlusion    sdfgi_read_sky_light    sdfgi_bounce_feedback    sdfgi_cascades    sdfgi_min_cell_size    sdfgi_cascade0_distance    sdfgi_max_distance    sdfgi_y_scale    sdfgi_energy    sdfgi_normal_bias    sdfgi_probe_bias    glow_enabled    glow_levels/1    glow_levels/2    glow_levels/3    glow_levels/4    glow_levels/5    glow_levels/6    glow_levels/7    glow_normalized    glow_intensity    glow_strength 	   glow_mix    glow_bloom    glow_blend_mode    glow_hdr_threshold    glow_hdr_scale    glow_hdr_luminance_cap    glow_map_strength 	   glow_map    fog_enabled    fog_light_color    fog_light_energy    fog_sun_scatter    fog_density    fog_aerial_perspective    fog_sky_affect    fog_height    fog_height_density    volumetric_fog_enabled    volumetric_fog_density    volumetric_fog_albedo    volumetric_fog_emission    volumetric_fog_emission_energy    volumetric_fog_gi_inject    volumetric_fog_anisotropy    volumetric_fog_length    volumetric_fog_detail_spread    volumetric_fog_ambient_inject    volumetric_fog_sky_affect -   volumetric_fog_temporal_reprojection_enabled ,   volumetric_fog_temporal_reprojection_amount    adjustment_enabled    adjustment_brightness    adjustment_contrast    adjustment_saturation    adjustment_color_correction           local://Sky_lg2ym Y	         local://Environment_wptfl m	         Sky             Environment                                RSRCr4�ײuhGST2   @   @      ����               @ @        �  RIFF�  WEBPVP8L�  /?����m��������_"�0@��^�"�v��s�}� �W��<f��Yn#I������wO���M`ҋ���N��m:�
��{-�4b7DԧQ��A �B�P��*B��v��
Q�-����^R�D���!(����T�B�*�*���%E["��M�\͆B�@�U$R�l)���{�B���@%P����g*Ųs�TP��a��dD
�6�9�UR�s����1ʲ�X�!�Ha�ߛ�$��N����i�a΁}c Rm��1��Q�c���fdB�5������J˚>>���s1��}����>����Y��?�TEDױ���s���\�T���4D����]ׯ�(aD��Ѓ!�a'\�G(��$+c$�|'�>����/B��c�v��_oH���9(l�fH������8��vV�m�^�|�m۶m�����q���k2�='���:_>��������á����-wӷU�x�˹�fa���������ӭ�M���SƷ7������|��v��v���m�d���ŝ,��L��Y��ݛ�X�\֣� ���{�#3���
�6������t`�
��t�4O��ǎ%����u[B�����O̲H��o߾��$���f���� �H��\��� �kߡ}�~$�f���N\�[�=�'��Nr:a���si����(9Lΰ���=����q-��W��LL%ɩ	��V����R)�=jM����d`�ԙHT�c���'ʦI��DD�R��C׶�&����|t Sw�|WV&�^��bt5WW,v�Ş�qf���+���Jf�t�s�-BG�t�"&�Ɗ����׵�Ջ�KL�2)gD� ���� NEƋ�R;k?.{L�$�y���{'��`��ٟ��i��{z�5��i������c���Z^�
h�+U�mC��b��J��uE�c�����h��}{�����i�'�9r�����ߨ򅿿��hR�Mt�Rb���C�DI��iZ�6i"�DN�3���J�zڷ#oL����Q �W��D@!'��;�� D*�K�J�%"�0�����pZԉO�A��b%�l�#��$A�W�A�*^i�$�%a��rvU5A�ɺ�'a<��&�DQ��r6ƈZC_B)�N�N(�����(z��y�&H�ض^��1Z4*,RQjԫ׶c����yq��4���?�R�����0�6f2Il9j��ZK�4���է�0؍è�ӈ�Uq�3�=[vQ�d$���±eϘA�����R�^��=%:�G�v��)�ǖ/��RcO���z .�ߺ��S&Q����o,X�`�����|��s�<3Z��lns'���vw���Y��>V����G�nuk:��5�U.�v��|����W���Z���4�@U3U�������|�r�?;�
 [remap]

importer="texture"
type="CompressedTexture2D"
uid="uid://do4v0552sgtv3"
path="res://.godot/imported/icon.png-487276ed1e3a0c39cad0279d744ee560.ctex"
metadata={
"vram_texture": false
}

[deps]

source_file="res://icon.png"
dest_files=["res://.godot/imported/icon.png-487276ed1e3a0c39cad0279d744ee560.ctex"]

[params]

compress/mode=0
compress/lossy_quality=0.7
compress/hdr_compression=1
compress/bptc_ldr=0
compress/normal_map=0
compress/channel_pack=0
mipmaps/generate=false
mipmaps/limit=-1
roughness/mode=0
roughness/src_normal=""
process/fix_alpha_border=true
process/premult_alpha=false
process/normal_map_invert_y=false
process/hdr_as_srgb=false
process/hdr_clamp_exposure=false
process/size_limit=0
detect_3d/compress_to=1
�)�����\׈l�@�GST2   �   �      ����               � �           RIFF   WEBPVP8L�  /��;G�����_��`BA�6L5��F�ݟ��m�j��L�?C�m��ps�@AM�l�A�K�bU��w�n�-f�����Ẅ��n��'1#w����h�  M�ڶm�i����C$�����kQv?G�_�d�uۀ qy�:"������n�������p��6������lym� �ǆ<gDZ�1Ԟ�4��!_%{<"{�=�ش��ɶ����!䲕ؾ����!�=JK��aH�/��i��u����r:?��8��I���?��l���D�U���@�~"�ku����́�+,��t�9&�2��}k�����'��aJi5Ή�ݖ����<�ƗƦ�8��[�M껾[�*|����>@��*�����ǭ�����[Hm��x���7�j���Lz�����NmU|�Sʏ{�_�6U�qn�Y�П��7�)���)#С�O���=ep�=v�iU@@e�� �x�1�-�' {(�-�PE���S�W����r����ǁH�*��{����?�oFn���Od:ejĺ͹m)���D�����VaV��[`�>�+����x�)�_ԶM��RU̐'��LXZI/��G�����L��������^R�J1��v|$~�,?gǴB�[�>��U�+��^�,���pԬ/�v�����\�y�)�.�Rڷ���~�~��>ا�/��]%6e�+���ᚧ���i:�۱�����lc��X��:����@u�}�-�}�&%�&˵m.}�:���86�";�iY�(���)>T��zQ���d�ǳ�2��#�dV�;�7�\�c�E�s[z��gm������s�]������4"��k'��T4xKN8�ն�/i������z����~�X�Uخ7" ��ʾ_���SY�����<�<���&�B6`CP� ���������_x��X�a�ǈ��;�P����K��=.f�b~��8ީZ9�IH(��l��Od�8c�fL��t��S�"�mj-���B���{�-q���js|�D@� o�d���?��6a����0���*c>��q���\dX� ���{�����+��+�+>�P^$04B�3���ѢE3
��'�?�os��H/|an=�Bn�p��)��k�Z:
}s��(��Պ�(�ؐ�qk��EN���7Y��GSa�_��N���/�=B{�>]wg�ެ�֧4{=V2����*�?��oX�)�i�t�;o��� xؐIU-C�.b>Ԟa�B"pF (��h�k����kl8�!����7�W9��PXj�N[m~Τ�a��C+ M�G�����|���?�8G"�r����qSo�;_�����"aSFȇ��x���F���p�9�����3��-0��J��	SZ�>Hϕ�(
���+�%���k9�V��
�P���L��mN��
��eD4�������l���)���hru;QE�h[y]�r_�J�
�ԋ|�΃ �B�mx���:��4�P��ɱ���7�9��D��7sP>����H�]���{��צa�C�zl[x�0��j5��N��2�5�Gk�f�B_�2Ƀ�6�	@Z�I���?��5��6f�w_dSF�����*�x�@Z E�J6NM��g�K�}ݸw�=a,��
bB3NX�
*pj[U��lrTPKJ�߄��\��2& �T�n�.�wl9���'x�mA6���-g<��8�k�s`�Z�C�I甚���V(;ÌBo��6�|=�ZraYD�s��(-ٕj��g�p�"�L}�_��q��6(���J����]�D���}�61!f�} ��?*�4���ڝ��HÒ�7\�ߘ��#[B�����P�H߬$'������M��*��	hJq��{Հ�+�@�[��P �P�+uW��W?ޱ����@����iY���ؐw��O�F�j�y�xGQ���#�E�+E����`��I����%�>.wtn�#���"s�s-�	LF��WA����\)��'�Z�S���L|�yH�4Pڶ֠�J2���.��'�hK��%f���@X+��n.6�(�K�x�6�n؃z枋��dO�o�sV�/��h��t./��G.�/1m"����6��K(dtW�օ�ۡP<n!,����Q��|�#�SQ��
2�nM��(���t�5k6�M� ^_̝X
E����Q�
�Mq�>���`�����$�ep���*mW�/ϱYkh�Ã����p���(%O\�a7ϸ�����7��TH/���)�0�I�����R��^��b�B���'�!K��,7�0,��s��Mgs0��>T�oӔ��m+ͨ9�2�^1J�/�7\as��U}n��k��S
e2+9��םQ��2�»��nk g#tWg�1�Y�*��Sܼ"a9x�v6隷���m��7�?��� ��0ZjN?���O���v�w}ݸXS��#&�U!5蝁	��;�b�^��==�IE���ͺj<���@[�>e��I�'MnqX���-���s��o]nû�Jװ��e��95�B�ǭ���s�c���ء`�"�;a��@�MU�+:��:���Ց��`k����G�&�ie�R��@������6��l-��b�}�B�;��iU��w\��iHu�����%���i	wY�������/�#W܅��(WM)Z
b�V]��[�O�_�ݯ�W�5��I����b1] �H�#k5�d���L�ћ-V�e:ܷI��b��LDT�&�۫��p���~��3��b�p�K����14�6�׮�O��M��x��/��+E:�^�;]��N^��-o����J�S akVj�%��R]�Nɛ����+l�cH�=
̅��U$L��а.,I��9�R��Q�1Ց�r�X(v埣
�bv�〹T�|N�c{�L �[T�w��ܗT��d���2���7%Y�4��.`u_`���1/���Th.�|�-�	w�a��X�&?e�"X��=]�)p)����-6˞�[`H�U/��8����P��kEW�Ѐ��ȹ����\-0X�$6�D��;�x�jG�V2"�4��_i�<����\Z.<9�U�O��Fg�:"�F�v�v��
�,2�r��#Ӧ�r�xݖ �&�ʉt=ٸ��.�D�,>�_6~[)�����L�PK<9H�X���H)s�^�Cg�ΦI�D=���t���9���,9J��&~V�)s9���si8I@PT�j��]�n����'Yw^��
�-�:����am��i�� ��9	$�Ġc��}O�>t�u�?p��"�	�r�fE�a�W-�G�
�����k��E�~���kI�/4ZB� d]�Ꞛ;g ����P��sǔ�=�qH'�	b�Js�vWh\<��3���2,F�dҺ�i�Rʾ���1Ih���*�T%�)p�z%��q|�J7Y���I���u��'�G�%6\��6�B�F`]M`�
�X����Gߺp��������?R*ee�(���]v(��f�>1l�!��������hRtA?�~4�:�QDڲ'/ˉ�|�EP�
�1�r�qM<�/d~�����&��kջ�x����u1�BY���̕f���h~��&�0
�+�w������-�*.Z��e��L�#�. MR�-"Y� ���<><5'�jK�B+�,D�\lQY׺D6ɣe���d��²VX����4KԐY�[��&Hjf��,�� �kWN~$y=obe���̒ݔ�'8�������p�T��s�'�ʢp��t
)��5Q��UL6�����`>�!̉T���ym���J��"��=$9+�BJ����Z!,vW��e�^(�q�Dޫ���Ҽ�@W���ƀz{s���x��)���MgQ>�������y?itQ���2��a!�'����zᆦ�-}�M����q򎄄����U���j���i�dV�g�����_!^�A�$����.@V��Sg�>ѥ�ɋK�����j�L�h�em�­Z�rm�ˆ� ڬ�(�r��1ոI�*+��f�O\�e�,�a�s��
�|i=	��O�L�eE
��t�6�z�5�Y�9r����=L��Z����Ш~�i�f��Sr�Lj���Di�KM),Lh2�i��@O\T~uO��;A����RxM�K�9�L��l)�|��3,�����1p��R閂ӵ�̦�֫��>�g���%�*Y��Zs��%�ܨB�V5?a�dr�c�
��Q�-v{s���y-]t&�����/�F��t�ӳ�T)�h�\��YZ(]���fs���N�v���&>��ܿ��N��ha�E3�*7���C���o!�n
�>ӕ�݅F����7!�I�b��J�i�L������j�3H�}���~�4����El*E�.,����u��CR��t�,T�?D��<ٯ_���\�t��e�s��N�*ϰK(w�4�_4�ф�$O�M�r\���Ҵ�H��D݅Z[ef�Y��d\�,��K��3����H�������S�T��O׀�d붨������$)���i�ip�|M��Dw; K�q�,rS��O��	3;���E'-UDg�0��'^��W��s��`��C�/�T�����iʠS/�����J�N�J��HS�	�CM���*˓��#���c-�ت���>]��&��4~B��Kk�����"��q���*$>�<�Ԗf�|�Pȁt���IZckY.9w1b��l��W��W�wܝZKMLSX�aas�����Z\�R���d��Z'��$ϩ�ך��ݧ)�����D(��EB~JR<[�2MfPɿ�$�2��� .C� ��9^�S���@�4��a��iR��MFM��i�;��:�����B�r�9��WU~���)X�|�H�\�5�����{a~�X�Y��M�Rޠ�W��
$��}�,���*���n{jۀQ 'rI���k�rP�W1�`�侔����A�v��F�J��n:�VHJš!dA��^���..�o���5L���:DD:^��9�-��s��H�q��Y�+�*�ǱlR]��6+8����v�S^�OW ��i�\�7�3�AF�ݞ6">�e!�.8���]RA��u@�;wE]0�~�5��][d�8u�0���i| �!����D����&Y����e%��l���M
�<�`ک=�#֘� ��b��l��1"��f(]e�U'L�IQ���*����f"�tS�9��s���x�w:���d�%
�������Y�`�<~�Tp��~�5#��U�a��ڕZՖ�(��	��h���a#��]����q�\���w~u�/��Q�0GE��u�L��c���]fnK;�q�qS?�u�h_ʂF�%\p�����쒰+��Y׳?ʹT|7�M��~�p���k"��%��4�u�#2�>�����m��de��.�^���XeR�r�%5�G�6�Q�`5w���9R�6YY�~3-0O�D�f��:�dxU����ɸK����rNbE��.�P�q��Te%7괙�¶k���͠n����C�b���ԭG��7��߭%[qO��,o"�IA
r�����p���ç*N�I��;�q|��u���e�wwp��ǻ�ks$�)-,��;��a���J�sW���w�P+H&}{��?�p����8���W�&�m���^�~p�թ��&�p�d���ЍL��,��>�f�x�Q���*����D�@�
�%���>�>����Э&e.��Q��6��mb�.KW;r�pm�t�Y �*�!�+H�*
�u1AߝLT�q.�M��
6� 4�4*dčWj���C?�X��_�憢�h�@�rt+&1%����S{��k���z�����I��0��pG��ӯ$K���i@�=B��Y��,��^�Rd�3�W����ry(�RKQ+G~~?:7|����~m޵��N��u!5�'�eL�� [�(�t���b�&�~L�$�nV�DG�Rэ�ʋ�~�>��/tV��+lKL)X�tT��=R�-��������A�u�{X����A�o����\�T�ӽFQS��3��������C.�D}��b�~K�fʸ4���D��������.&Y��7
�+s����B����/�m�_&���g{"������J����^��+��[�gn��D����t�OBzw[�G�g�� ׬��:r��^���^+%9L��H1$�LP;w�'b@no�kIx�YSM?��XA��Yi�*�7l60qw[�ٔQ��iح�ጓ0[���D�6���t�Y���Tz��a�h~?��-�t�{�w����K;�a�*4$�5�wXn>�[�A�[� 5�B�h�:^rC�+H����C��Ӹ�Y���p�S��Dg�Z'��m!RW)ц��V�	<��1��UjyX�OΖ�sCp���Y�|���
4���>:@�w���>�kFL{)[�F�Co5(��BC��r�h�y����r�gd�n�T!�+�Q�Ö��J�%�[��o��� ����"{�Э�E&S.��(n��d ��C�_�7f2jmU�(�*7[�Y0���д�֛� pK��:b�3P��HU�3��2��BF�2]�h����SР5�x�-z���c�Q�}�ƶt�;���Y��k���B��:�0�QdcE��!{Y�sP�8Z���skVȒ6�@L.ՏTb���𺐴��'�f���j�������a�Od1Gbr�+��D
{i��T;-�>�ʡ�,M ڼPzu8#D�$�@,��L��0&�K�c�{hP-�����v��^��@\�2���Ԛ���Y�U(�bQ/#�PmH��d�6n�m^��-v8Kk���RFb^!�v��4�e�⌣�ۢ�h�(`��/��%�E�5�3�p�~�*�������X��}-X�kf��4�Ⱥ�jaY9��=����R�0��<�����6\�Ϟ�5�J��FZ p�j%F?��}��{��4��v[����}V��.�:\�K�>�/d���#a8�a� ^���+l�HL  �Sc��������H�6�ΆC �=��r�M{ï�&ƫ�e6��Ũ�?)0�c0��� �)���IgA��lIX�jC/�����E�.�R$�QS:��/X���Y`��7�d�\�q�N�@�6�
 ��],��{{1@�^���+��d�r�5톞 ��#��VR'�?��9��#�4Af=ڗ���������6��BX*<?Y����c�*����,v�_��ks�@ND:! A@���R��N��q�Q�6t�i��B��0�����v��+31���j�X��0��B��c���3�o7qT�2�����O�/�m�Z�RrIG\:e�.!�#�4&�����䋨���rim��(u�O�?��#�������P5t�_�먘d�6
����DOr�y�yi,dֳ���������W���kY���:�L�?˰�}�Y�n����}���hۮ��ڈIk����o���zG����V��Ѿfֱ���B�	�i��j#�C�1n���b�/��dza� ^ ���!�6{?a
+OKlw�ӏ�f)�C˰�:y�g��ػ��Czƴ��edEJ
Ho�J�m/��7���������=�_7�I����ks�q�A1|Yp�Zg�Ǒ���;�#�ޕ�p���ΐ����!�ʿ�����8�d+:v����
A�:>��Xɩ��h��m��h1tK?���G�� ����C:��킄>T�s�:�m}S�i��� U�-�7!���� ��v��0m�ׂ`h,�^�|�K��)	�?�6��W�W�DỴ�L'��V2J=�M�'���	=$�u�s�#���d� 	��|���T!�O�,C�/���XD�4a���~���sxpk���C�!��p<mk�;&��+�x�o�+>B}��T��d���|�m@��+s<v�}��=pD�/�1��)O���}��C{��'߈&���t���A�[�WMA���	'���H�'H_�>�� ���	�z��п�+���q<���X��o�֗~f���#	���#<��$��V� yG������?�Vߜ-u�6�	�?W�:���U�z�/ q��+G,�=U[remap]

importer="texture"
type="CompressedTexture2D"
uid="uid://dlyw6fyas1sr8"
path="res://.godot/imported/light_mask.png-40da3c93e1795f65c34ad69a6ae38ba3.ctex"
metadata={
"vram_texture": false
}

[deps]

source_file="res://light_mask.png"
dest_files=["res://.godot/imported/light_mask.png-40da3c93e1795f65c34ad69a6ae38ba3.ctex"]

[params]

compress/mode=0
compress/lossy_quality=0.7
compress/hdr_compression=1
compress/bptc_ldr=0
compress/normal_map=0
compress/channel_pack=0
mipmaps/generate=false
mipmaps/limit=-1
roughness/mode=0
roughness/src_normal=""
process/fix_alpha_border=true
process/premult_alpha=false
process/normal_map_invert_y=false
process/hdr_as_srgb=false
process/hdr_clamp_exposure=false
process/size_limit=0
detect_3d/compress_to=1
� �����m�,�GST2   F   F      ����               F F        �  RIFF�  WEBPVP8L�  /E@�㦑$I���;����:ͣ�8�$�In���߾f�AI��M�<�O`�{D�-$��`��V�	6����?�4,���Q�Z����Ҹv���Qv\�Ea�i�V�Ƅ;KT0�m�H��##�3��4��ӗ%�6l�Ft:%���+T�@_F~�W�JR���V�
����&���f�Ɣ��d������Y#��i|�n���?�:c��9�%�8��y�~������.PԶmm�H*s�y�e���2�3�Q�7�Z"'��⌵����KG������=�P���jY�Z��n��V��]�ǽAŭ�|�(9MYn"I-���6��Wj��"5I [��9����i��� �JjJ,Ԑ��ZI�xE(he�t�zs�kjÀ=�'nQkH��C����+���ɚH����F�"c�7ڣ^c4Q�`<j�-�<(��N��Td��Fg��rH�=���4�+`�kZs!Q�Ѫ �mH�����Z�Ɍ��j�R�yy���svT{	xs(g�[�Ao�CYV̼�Mg���i.��ƾm�������]`O�?¼�ZV@��	��P�V��E�L�P�#�e��+����A��Q&iQ-�Hp�J9�$}l�%T����)�5�5��K��� �j��p�:�=�n�B]�-�&=H"��V2�íE�J���j{)iIzc���EJEL/��z�w��Ś�"�lx�"���s�=����Г�	&Iw��5�ћ�&����gnyn,��lɇ%D�1�������*2L�#��O����D���~Q��lr��㯷�?���BS�;f������h </��cdJ
D��{�P}F���@Q����s�PQL�<"�w0��w�IF�R�ːc���{xf5�q~~�M�a��WC��=�͍���L����A��=����U�ZOQ�E��f�P@��+�*�,)�F(\'%6I��$pG��}�%[=�r|���W�лmb������&"T$�J�Ф˽����x��|C�Y�Ŕ��ˈ";�D,Չ���9L�����#�KTm����ə�YdF��[��qE�Cke��9��qGL��䐘>s	����ӕ~�F�Md�<��{	xR� &��h~*���g��~	��P��t��&��D��P���H()F�aN66$�y��3P�ȟ����f�Nu�	s�XB[���f�K*�Ќ|HGE��Wu���ЯP��[�^)�
;���|��bހ+TE
�}z.����'�Q�,	J����uM׹1�<S%r�;��LC�W5O6�r��)݃o���4+	F^�F}�R �R���F�e�鉬t����i��C7%#/f� Mv7K;ٚ�f������������������Mh#�Э��l��|s�\�-�)J3����9�ME:�z`�T�㳋d��4-�问�Q�<>���>�J 7%}�����������˷ݍ�=�5R [remap]

importer="texture"
type="CompressedTexture2D"
uid="uid://cy4acafft8vjv"
path="res://.godot/imported/lollipopGreen.png-ad04020a819959359f29965dcb47712e.ctex"
metadata={
"vram_texture": false
}

[deps]

source_file="res://lollipopGreen.png"
dest_files=["res://.godot/imported/lollipopGreen.png-ad04020a819959359f29965dcb47712e.ctex"]

[params]

compress/mode=0
compress/lossy_quality=0.7
compress/hdr_compression=1
compress/bptc_ldr=0
compress/normal_map=0
compress/channel_pack=0
mipmaps/generate=false
mipmaps/limit=-1
roughness/mode=0
roughness/src_normal=""
process/fix_alpha_border=true
process/premult_alpha=false
process/normal_map_invert_y=false
process/hdr_as_srgb=false
process/hdr_clamp_exposure=false
process/size_limit=0
detect_3d/compress_to=1
��Nextends CharacterBody2D


# Declare member variables here. Examples:
# var a = 2
# var b = "text"
const pad = 100
const WIDTH = 1280
const HEIGHT = 720
const MAX_FRUIT = 10
var _bounds := Rect2(pad,pad,WIDTH-2*pad,HEIGHT-2*pad)

@export var speed := 500
@export var friction = 0.18
var _velocity := Vector2.ZERO
var _action = Vector2.ZERO
var _heuristic = "player"
@onready var fruit = $"../Fruit"
@onready var raycast_sensor = $"RaycastSensor2D"
@onready var walls := $"../Walls"
@onready var colision_shape := $"CollisionShape2D"
var fruit_just_entered = false
var just_hit_wall = false
var done = false
var best_fruit_distance = 10000.0
var fruit_count = 0
var n_steps = 0
var MAX_STEPS = 20000
var needs_reset = false
var reward = 0.0

func _ready():
	raycast_sensor.activate()
	reset()

func _physics_process(delta):
	n_steps +=1    
	if n_steps >= MAX_STEPS:
		done = true
		needs_reset = true

	if needs_reset:
		needs_reset = false
		reset()
		return
	
	var direction = get_direction()
	if direction.length() > 1.0:
		direction = direction.normalized()
	# Using the follow steering behavior.
	var target_velocity = direction * speed
	_velocity += (target_velocity - _velocity) * friction
	set_velocity(_velocity)
	move_and_slide()
	_velocity = velocity
	
	update_reward()
		
	if Input.is_action_just_pressed("r_key"):
		reset()

func reset():
	needs_reset = false
	fruit_just_entered = false
	just_hit_wall = false
	#done = false
	fruit_count = 0
	_velocity = Vector2.ZERO
	_action = Vector2.ZERO
	position = _calculate_new_position()
	spawn_fruit()
	
#    position.x = randf_range(_bounds.position.x, _bounds.end.x)
#    position.y = randf_range(_bounds.position.y, _bounds.end.y)	
#    fruit.position.x = randf_range(_bounds.position.x, _bounds.end.x)
#    fruit.position.y = randf_range(_bounds.position.y, _bounds.end.y)
	best_fruit_distance = position.distance_to(fruit.position)
	n_steps = 0 

func _calculate_new_position(position: Vector2=Vector2.ZERO) -> Vector2:
	var new_position := Vector2.ZERO
	new_position.x = randf_range(_bounds.position.x, _bounds.end.x)
	new_position.y = randf_range(_bounds.position.y, _bounds.end.y)	
	
	if (position - new_position).length() < 4.0*colision_shape.shape.get_radius():
		return _calculate_new_position(position)

	var radius = colision_shape.shape.get_radius()
	var rect = Rect2(new_position-Vector2(radius, radius), 
	Vector2(radius*2, radius*2)
	)    
	for wall in walls.get_children():
		#wall = wall as Area2D
		var cr = wall.get_node("ColorRect")
		var rect2 = Rect2(cr.get_position()+wall.position, cr.get_size())
		if rect.intersects(rect2):
			return _calculate_new_position()
	
	return new_position
	
	
func get_direction():
	if done:
		_velocity = Vector2.ZERO
		return Vector2.ZERO
		
	if _heuristic == "model":
		return _action
		
	var direction := Vector2(
		Input.get_action_strength("move_right") - Input.get_action_strength("move_left"),
		Input.get_action_strength("move_down") - Input.get_action_strength("move_up")
	)
	
	return direction
	
func set_action(action):
	_action.x = action["move"][0]
	_action.y = action["move"][1]
	
func reset_if_done():
	if done:
		reset()
		
func get_obs():
	var relative = fruit.position - position
	var distance = relative.length() / 1500.0 
	relative = relative.normalized() 
	var result := []
	result.append(((position.x / WIDTH)-0.5) * 2)
	result.append(((position.y / HEIGHT)-0.5) * 2)  
	result.append(relative.x)
	result.append(relative.y)
	result.append(distance)
	var raycast_obs = raycast_sensor.get_observation()
	result.append_array(raycast_obs)

	return {
		"obs": result,
	}
	
	
func update_reward():
	reward -= 0.01 # step penalty
	reward += shaping_reward()
	
func zero_reward():
	reward = 0.0   

func get_reward():
	return reward

	
func shaping_reward():
	var s_reward = 0.0
	var fruit_distance = position.distance_to(fruit.position)
	
	if fruit_distance < best_fruit_distance:
		s_reward += best_fruit_distance - fruit_distance
		best_fruit_distance = fruit_distance
		
	s_reward /= 100.0
	return s_reward
	
func set_heuristic(heuristic):
	self._heuristic = heuristic

func get_obs_space():
	# typs of obs space: box, discrete, repeated
	return {
		"obs": {
			"size": [len(get_obs()["obs"])],
			"space": "box"
 }
 }
	
func get_action_space():
	return {
		"move" : {
		"size": 2,	
			"action_type": "continuous"
 }
 }   

func get_done():
	return done
	
func set_done_false():
	done = false

func spawn_fruit():
	fruit.position = _calculate_new_position(position)
	best_fruit_distance = position.distance_to(fruit.position)

func fruit_collected():
	fruit_just_entered = true
	reward += 10.0
	fruit_count += 1
	spawn_fruit()
#    if fruit_count > MAX_FRUIT:
#        done = true

	
func wall_hit():
	done = true
	reward -= 10.0
	just_hit_wall = true
	reset()

func _on_Fruit_body_entered(body):
	fruit_collected()


func _on_LeftWall_body_entered(body):
	wall_hit()
func _on_RightWall_body_entered(body):
	wall_hit()
func _on_TopWall_body_entered(body):
	wall_hit()
func _on_BottomWall_body_entered(body):
	wall_hit()
�\yi��[remap]

path="res://.godot/exported/133200997/export-5127fc3fc2907f1e87ac8e869558593d-ExampleRaycastSensor2D.scn"
z;Aغ����+[remap]

path="res://.godot/exported/133200997/export-7454c66d21916090bf5dc7766fa8629a-RaycastSensor2D.scn"
��Q^[remap]

path="res://.godot/exported/133200997/export-1222b1ceeacbc57c3f8fe0e4235d9be0-ExampleRaycastSensor3D.scn"
ֵ�KPȗo1�![remap]

path="res://.godot/exported/133200997/export-1cc1ad7ce1f98ed0c4cee4b060bc26fd-RaycastSensor3D.scn"
��ܯ[remap]

path="res://.godot/exported/133200997/export-d1d9cc1906d18d9a32f853b74a4d90f4-RGBCameraSensor3D.scn"
;�[remap]

path="res://.godot/exported/133200997/export-f42ac285715e233e30405f3a6a1edaf9-BallChase.scn"
�,�����J�[remap]

path="res://.godot/exported/133200997/export-e0c93c3e5af95d0922ba36d1569eeaa1-BatchEnvs.scn"
 l�De^�u=[remap]

path="res://.godot/exported/133200997/export-7cf3fd67ad9f55210191d77b582b8209-default_env.res"
s���"'v�PNG

   IHDR   @   @   �iq�   sRGB ���  �IDATx��ytTU��?�ի%���@ȞY1JZ �iA�i�[P��e��c;�.`Ow+4�>�(}z�EF�Dm�:�h��IHHB�BR!{%�Zߛ?��	U�T�
���:��]~�������-�	Ì�{q*�h$e-
�)��'�d�b(��.�B�6��J�ĩ=;���Cv�j��E~Z��+��CQ�AA�����;�.�	�^P	���ARkUjQ�b�,#;�8�6��P~,� �0�h%*QzE� �"��T��
�=1p:lX�Pd�Y���(:g����kZx ��A���띊3G�Di� !�6����A҆ @�$JkD�$��/�nYE��< Q���<]V�5O!���>2<��f��8�I��8��f:a�|+�/�l9�DEp�-�t]9)C�o��M~�k��tw�r������w��|r�Ξ�	�S�)^� ��c�eg$�vE17ϟ�(�|���Ѧ*����
����^���uD�̴D����h�����R��O�bv�Y����j^�SN֝
������PP���������Y>����&�P��.3+�$��ݷ�����{n����_5c�99�fbסF&�k�mv���bN�T���F���A�9�
(.�'*"��[��c�{ԛmNު8���3�~V� az
�沵�f�sD��&+[���ke3o>r��������T�]����* ���f�~nX�Ȉ���w+�G���F�,U�� D�Դ0赍�!�B�q�c�(
ܱ��f�yT�:��1�� +����C|��-�T��D�M��\|�K�j��<yJ, ����n��1.FZ�d$I0݀8]��Jn_� ���j~����ցV���������1@M�)`F�BM����^x�>
����`��I�˿��wΛ	����W[�����v��E�����u��~��{R�(����3���������y����C��!��nHe�T�Z�����K�P`ǁF´�nH啝���=>id,�>�GW-糓F������m<P8�{o[D����w�Q��=N}�!+�����-�<{[���������w�u�L�����4�����Uc�s��F�륟��c�g�u�s��N��lu���}ן($D��ת8m�Q�V	l�;��(��ڌ���k�
s\��JDIͦOzp��مh����T���IDI���W�Iǧ�X���g��O��a�\:���>����g���%|����i)	�v��]u.�^�:Gk��i)	>��T@k{'	=�������@a�$zZ�;}�󩀒��T�6�Xq&1aWO�,&L�cřT�4P���g[�
p�2��~;� ��Ҭ�29�xri� ��?��)��_��@s[��^�ܴhnɝ4&'
��NanZ4��^Js[ǘ��2���x?Oܷ�$��3�$r����Q��1@�����~��Y�Qܑ�Hjl(}�v�4vSr�iT�1���f������(���A�ᥕ�$� X,�3'�0s����×ƺk~2~'�[�ё�&F�8{2O�y�n�-`^/FPB�?.�N�AO]]�� �n]β[�SR�kN%;>�k��5������]8������=p����Ցh������`}�
�J�8-��ʺ����� �fl˫[8�?E9q�2&������p��<�r�8x� [^݂��2�X��z�V+7N����V@j�A����hl��/+/'5�3�?;9
�(�Ef'Gyҍ���̣�h4RSS� ����������j�Z��jI��x��dE-y�a�X�/�����:��� +k�� �"˖/���+`��],[��UVV4u��P �˻�AA`��)*ZB\\��9lܸ�]{N��礑]6�Hnnqqq-a��Qxy�7�`=8A�Sm&�Q�����u�0hsPz����yJt�[�>�/ޫ�il�����.��ǳ���9��
_
��<s���wT�S������;F����-{k�����T�Z^���z�!t�۰؝^�^*���؝c
���;��7]h^
��PA��+@��gA*+�K��ˌ�)S�1��(Ե��ǯ�h����õ�M�`��p�cC�T")�z�j�w��V��@��D��N�^M\����m�zY��C�Ҙ�I����N�Ϭ��{�9�)����o���C���h�����ʆ.��׏(�ҫ���@�Tf%yZt���wg�4s�]f�q뗣�ǆi�l�⵲3t��I���O��v;Z�g��l��l��kAJѩU^wj�(��������{���)�9�T���KrE�V!�D���aw���x[�I��tZ�0Y �%E�͹���n�G�P�"5FӨ��M�K�!>R���$�.x����h=gϝ�K&@-F��=}�=�����5���s �CFwa���8��u?_����D#���x:R!5&��_�]���*�O��;�)Ȉ�@�g�����ou�Q�v���J�G�6�P�������7��-���	պ^#�C�S��[]3��1���IY��.Ȉ!6\K�:��?9�Ev��S]�l;��?/� ��5�p�X��f�1�;5�S�ye��Ƅ���,Da�>�� O.�AJL(���pL�C5ij޿hBƾ���ڎ�)s��9$D�p���I��e�,ə�+;?�t��v�p�-��&����	V���x���yuo-G&8->�xt�t������Rv��Y�4ZnT�4P]�HA�4�a�T�ǅ1`u\�,���hZ����S������o翿���{�릨ZRq��Y��fat�[����[z9��4�U�V��Anb$Kg������]������8�M0(WeU�H�\n_��¹�C�F�F�}����8d�N��.��]���u�,%Z�F-���E�'����q�L�\������=H�W'�L{�BP0Z���Y�̞���DE��I�N7���c��S���7�Xm�/`�	�+`����X_��KI��^��F\�aD�����~�+M����ㅤ��	SY��/�.�`���:�9Q�c �38K�j�0Y�D�8����W;ܲ�pTt��6P,� Nǵ��Æ�:(���&�N�/ X��i%�?�_P	�n�F�.^�G�E���鬫>?���"@v�2���A~�aԹ_[P, n��N������_rƢ��    IEND�B`��ԍ�q{   �����)K   res://addons/godot_rl_agents/sensors/sensors_3d/ExampleRaycastSensor3D.tscn���IH�@%   res://addons/godot_rl_agents/icon.png�D!R��*   res://BallChase.tscnǅ1��?   res://BatchEnvs.tscn)�E4WnUb   res://cherry.png�M�>�q   res://icon.png�
h���n   res://light_mask.png���qZ   res://lollipopGreen.png�w�+,�:r��ECFG      _global_script_classes�                    base      Node2D        class      	   ISensor2D         language      GDScript      path   <   res://addons/godot_rl_agents/sensors/sensors_2d/ISensor2D.gd            base      Node3D        class      	   ISensor3D         language      GDScript      path   <   res://addons/godot_rl_agents/sensors/sensors_3d/ISensor3D.gd            base      Node3D        class         RGBCameraSensor3D         language      GDScript      path   D   res://addons/godot_rl_agents/sensors/sensors_3d/RGBCameraSensor3D.gd            base   	   ISensor3D         class         RayCastSensor3D       language      GDScript      path   B   res://addons/godot_rl_agents/sensors/sensors_3d/RaycastSensor3D.gd              base   	   ISensor2D         class         RaycastSensor2D       language      GDScript      path   B   res://addons/godot_rl_agents/sensors/sensors_2d/RaycastSensor2D.gd     _global_script_class_icons�            	   ISensor2D             	   ISensor3D                RGBCameraSensor3D                RayCastSensor3D              RaycastSensor2D           application/config/name      	   BallChase      application/run/main_scene         res://BatchEnvs.tscn   application/config/features   "         4.0    application/config/icon         res://icon.png     display/window/stretch/mode         2d     display/window/size/width            display/window/size/height      �     editor_plugins/enabled   "          input/move_left�              deadzone      ?      events              InputEventKey         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          pressed           keycode           physical_keycode   A      unicode           echo          script         input/move_right�              deadzone      ?      events              InputEventKey         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          pressed           keycode           physical_keycode   D      unicode           echo          script         input/move_up�              deadzone      ?      events              InputEventKey         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          pressed           keycode           physical_keycode   W      unicode           echo          script         input/move_down�              deadzone      ?      events              InputEventKey         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          pressed           keycode           physical_keycode   S      unicode           echo          script         input/left_arrow�              deadzone      ?      events              InputEventKey         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          pressed           keycode           physical_keycode    @    unicode           echo          script         input/right_arrow�              deadzone      ?      events              InputEventKey         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          pressed           keycode           physical_keycode    @    unicode           echo          script         input/up_arrow�              deadzone      ?      events              InputEventKey         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          pressed           keycode           physical_keycode    @    unicode           echo          script         input/down_arrow�              deadzone      ?      events              InputEventKey         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          pressed           keycode           physical_keycode    @    unicode           echo          script         input/r_key�              deadzone      ?      events              InputEventKey         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          pressed           keycode           physical_keycode   R      unicode           echo          script         input/zoom_out�              deadzone      ?      events              InputEventMouseButton         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          button_mask           position              global_position               factor       �?   button_index         pressed           double_click          script         input/zoom_in�              deadzone      ?      events              InputEventMouseButton         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          button_mask           position              global_position               factor       �?   button_index         pressed           double_click          script         input/reset_camera�              deadzone      ?      events              InputEventMouseButton         resource_local_to_scene           resource_name             device     ����	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          button_mask           position              global_position               factor       �?   button_index         pressed           double_click          script      )   physics/common/enable_pause_aware_picking            rendering/quality/filters/msaa         )   rendering/environment/default_environment          res://default_env.tres  �^�@�*��R�(~_