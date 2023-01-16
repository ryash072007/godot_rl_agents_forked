extends CharacterBody3D

const MOVE_SPEED = 12
const JUMP_FORCE = 30
const GRAVITY = 0.98
const MAX_FALL_SPEED = 30
const TURN_SENS = 2.0
const MAX_STEPS = 20000

@onready var cam = $Camera3D
var move_vec = Vector3()
var y_velo = 0
var needs_reset = false
# RL related variables
@onready var end_position = $"../EndPosition"
@onready var raycast_sensor = $"RayCastSensor3D"
@onready var first_jump_pad = $"../Pads/FirstPad"
@onready var second_jump_pad = $"../Pads/SecondPad"
@onready var robot = $Robot

var next = 1
var done = false
var just_reached_end = false
var just_reached_next = false
var just_fell_off = false
var best_goal_distance := 10000.0
var grounded := false
var _heuristic := "player"
var move_action := 0.0
var turn_action := 0.0
var jump_action := false
var n_steps = 0
var _goal_vec = null
var reward = 0.0

func _ready():
	raycast_sensor.activate()
	reset()

#func _process(_delta):
#	if _goal_vec != null:
#		DebugDraw.draw_line_3d(position, position + (_goal_vec*10), Color(1, 1, 0))

func _physics_process(_delta):
	
	#reward = 0.0
	n_steps +=1    
	if n_steps >= MAX_STEPS:
		done = true
		needs_reset = true

	if needs_reset:
		needs_reset = false
		reset()
		return
	
	
	move_vec *= 0
	move_vec = get_move_vec()
	#move_vec = move_vec.normalized()
	move_vec = move_vec.rotated(Vector3(0, 1, 0), rotation.y)
	move_vec *= MOVE_SPEED
	move_vec.y = y_velo
	set_velocity(move_vec)
	set_up_direction(Vector3(0, 1, 0))
	move_and_slide()
	
	# turning
	
	var turn_vec = get_turn_vec()
	rotation.y += deg_to_rad(turn_vec*TURN_SENS)
 
	grounded = is_on_floor()

	y_velo -= GRAVITY
	var just_jumped = false
	if grounded and get_jump_action():
		robot.set_animation("jump")
		just_jumped = true
		y_velo = JUMP_FORCE
		grounded = false
	if grounded and y_velo <= 0:
		y_velo = -0.1
	if y_velo < -MAX_FALL_SPEED:
		y_velo = -MAX_FALL_SPEED
	
	if y_velo < 0 and !grounded :
		robot.set_animation("falling")
	
	var horizontal_speed = Vector2(move_vec.x, move_vec.z)
	if horizontal_speed.length() < 0.1 and grounded:
		robot.set_animation("idle")
	elif horizontal_speed.length() < 1.0 and grounded:
		robot.set_animation("walk")    
	elif horizontal_speed.length() >= 1.0 and grounded:
		robot.set_animation("run")
	
	update_reward()
	
	if Input.is_action_just_pressed("r_key"):
		reset()
		

func get_move_vec() -> Vector3:
	if done:
		move_vec = Vector3.ZERO
		return move_vec
	
	if _heuristic == "model":
		return Vector3(
		0,
		0,
		clamp(move_action, -1.0, 0.5)
	)
		
	var move_vec := Vector3(
		0,
		0,
		clamp(Input.get_action_strength("move_backwards") - Input.get_action_strength("move_forwards"),-1.0, 0.5)
		
	)
	return move_vec

func get_turn_vec() -> float:
	if _heuristic == "model":
		return turn_action
	var rotation_amount = Input.get_action_strength("turn_left") - Input.get_action_strength("turn_right")

	return rotation_amount
	
func get_jump_action() -> bool:
	if done:
		jump_action = false
		return jump_action
		
	if _heuristic == "model":
		return jump_action  
	
	return Input.is_action_just_pressed("jump")
  
func reset():
	needs_reset = false
	next = 1
	n_steps = 0
	first_jump_pad.position = Vector3.ZERO
	second_jump_pad.position = Vector3(0,0,-12)
	just_reached_end = false
	just_fell_off = false
	jump_action = false

	set_position(Vector3(0,5,0))
	rotation.y = deg_to_rad(randf_range(-180,180))
	y_velo = 0.1
	reset_best_goal_distance()
	
func set_action(action):
	move_action = action["move"][0]
	turn_action = action["turn"][0]
	jump_action = action["jump"] == 1
	
func reset_if_done():
	if done:
		reset()

func get_obs():
	var goal_distance = 0.0
	var goal_vector = Vector3.ZERO
	if next == 0:
		goal_distance = position.distance_to(first_jump_pad.position)
		goal_vector = (first_jump_pad.position - position).normalized()
		
	if next == 1:
		goal_distance = position.distance_to(second_jump_pad.position)
		goal_vector = (second_jump_pad.position - position).normalized()
	
	goal_vector = goal_vector.rotated(Vector3.UP, -rotation.y)
	
	goal_distance = clamp(goal_distance, 0.0, 20.0)
	var obs = []
	obs.append_array([move_vec.x/MOVE_SPEED,
					move_vec.y/MAX_FALL_SPEED,
					move_vec.z/MOVE_SPEED])
	obs.append_array([goal_distance/20.0,
					goal_vector.x, 
					goal_vector.y, 
					goal_vector.z])
	obs.append(grounded)
	obs.append_array(raycast_sensor.get_observation())
	
	return {
		"obs": obs,
	}
	
func get_obs_space():
	# typs of obs space: box, discrete, repeated
	return {
		"obs": {
			"size": [len(get_obs()["obs"])],
			"space": "box"
			}
		}
	
func update_reward():
	reward -= 0.01 # step penalty
	reward += shaping_reward()
	
func get_reward():
	var current_reward = reward
	reward = 0 # reset the reward to zero checked every decision step
	return current_reward
	
func shaping_reward():
	var s_reward = 0.0
	var goal_distance = 0
	if next == 0:
		goal_distance = position.distance_to(first_jump_pad.position)
	if next == 1:
		goal_distance = position.distance_to(second_jump_pad.position)
	#print(goal_distance)
	if goal_distance < best_goal_distance:
		s_reward += best_goal_distance - goal_distance
		best_goal_distance = goal_distance
		
	s_reward /= 1.0
	return s_reward   

func reset_best_goal_distance():
	if next == 0:
		best_goal_distance = position.distance_to(first_jump_pad.position)
	if next == 1:
		best_goal_distance = position.distance_to(second_jump_pad.position)    

func set_heuristic(heuristic):
	self._heuristic = heuristic

func get_obs_size():
	return len(get_obs())
	
func zero_reward():
	reward = 0
   
func get_action_space():
	return {
		"move" : {
			"size": 1,
			"action_type": "continuous"
		},        
			"turn" : {
			"size": 1,
			"action_type": "continuous"
		},
		"jump": {
			"size": 2,
			"action_type": "discrete"
			}
		}

func get_done():
	return done
	
func set_done_false():
	done = false

func calculate_translation(other_pad_translation : Vector3) -> Vector3:
	var new_translation := Vector3.ZERO
	var distance = randf_range(12,16)
	var angle = randf_range(-180,180)
	new_translation.z = other_pad_translation.z + sin(deg_to_rad(angle))*distance 
	new_translation.x = other_pad_translation.x + cos(deg_to_rad(angle))*distance
	
	return new_translation


func _on_First_Pad_Trigger_body_entered(body):
	if next != 0:
		return
	reward += 100.0
	next = 1
	reset_best_goal_distance()
	second_jump_pad.position = calculate_translation(first_jump_pad.position)

func _on_Second_Trigger_body_entered(body):
	if next != 1:
		return
	reward += 100.0
	next = 0
	reset_best_goal_distance()
	first_jump_pad.position = calculate_translation(second_jump_pad.position)
		


func _on_ResetTriggerBox_body_entered(body):
	done = true
	reset()
