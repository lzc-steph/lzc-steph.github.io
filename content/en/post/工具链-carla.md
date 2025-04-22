---
date: 2025-04-19T04:00:59-07:00
description: ""
featured_image: "/images/carla/pia.jpg"
tags: ["automatic driving"]
title: "工具链-carla"
---

## 0 预备

下载 carla

1. ﻿﻿﻿Get Python from anaconda.com

2. ﻿﻿﻿Create virtual environment

   - ﻿﻿launch anaconda prompt

   - ﻿﻿conda create --name carla-sim python=3.7

   - ﻿﻿activate carla-sim

3. Install additional libraries via "pip install":
   + ﻿﻿pip install carla, pygame, numpy, jupyter and opencv-python

&nbsp;

## 1 基本构件块

```python
# all imports
import carla 				# the sim Library itself 
import random 			# to pick random spawn point 
import cv2 					# to work with images from cameras
import numpy as np 	# in this example to change image representation - re-shaping
```

```python
# connect to the sim
client = carla.Client('localhost', 2000)
```

<!--more-->

```python
# optional to Load different towns
client.load_world('Towne5')
```

```python
# define environment/world and get possible places to spawn a car
world = client.get_world()
spawn_points = world.get_map().get_spawn_points()
```

```python
# Look for a blueprint of Mini car,得到很多名为"mini"的汽车实例
vehicle_bp = world.get_blueprint_library().filter('*mini*')
```

```python
# spawn a car in a random location
start_point = random.choice(spawn_points)
vehicle = world.try_spawn_actor(vehicle_bp[0], start_point)
```

```python
# move simulator view to the car
spectator = world.get_spectator()
start_point.location.z = start_point.location.z+1 	#start_point was used to spawn
spectator.set_transform(start_point)
```

```python
#send the car off on autopilot - this will leave the spectator
vehicle.set_autopilot(True)
```

&nbsp;

## 2 在汽车上设置摄像头

```python
#setting RGB Camera - this follow the approach explained in a Carla video
# Link: https://www.youtube.com/watch?v=om8kLsBj4rc&t=1184s

#camera mount offset on the car - you can tweak these to each car to avoid any p
CAMERA_POS_Z = 1.6 																# this means 1.6m up from the ground
CAMERA_POS_X = 0.9 																# this is 0.9m forward

camera_op = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640') 		# this ratio works in CARLA 9.14 
camera_bp.set_attribute('image_size_y', '360')

camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z, x=CAMERA_POS_X))
# this creates the camera in the sim
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

def camera_callback(image, data _dict):
  data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
                                    
image_w = camera_bp.get_attribute('image_size_x') .as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()
                    
camera_data = {'image': np.zeros ((image_h, image_w,4))}
# this actually opens a live stream from the camera
camera.listen(lambda image: camera_callback(image, camera_data))
```

```python
# grab a snapshot from the camera an show in a pop-up window
img = camera_data['image']
cv2.imshow('RGB Camera', img)
cv2.waitKey(0)
```

Output: -1

```python
# clean up after yourself

camera.stop() # this is the opposite of camera.listen
for actor in world.get_actors().filter('*vehicle*'):
  	actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
  	sensor.destroy()
```

&nbsp;

&nbsp;

## 3 控制汽车

### 速度和油门踏板

```python
'''
This is the new Bit for tutorial 4
First we need to create controls functions so we could
push the car along the route
'''
# define speed contstants
PREFERRED_SPEED = 20 			# what it says
SPEED_THRESHOLD = 2 			# defines when we get close to desired speed so we drop the

# adding params to display text to image
font = CV2.FONT_HERSHEY_SIMPLEX
# org - defining lines to display telemetry values on the screen
org = (30, 30) 						# this Line will be used to show current speed
org2 = (30, 50) 					# this Line will be used for future steering angle
org3 = (30, 70) 					# and another line for future telemetry outputs
fontScale = 0.5
# white color
color = (255, 255, 255)
# line thickness of 2 px
thickness = 1

def maintain_ speed(s):
  '''
  this is a very simple function to maintan desired speed s arg is actual current speed
  '''
	if 5 >= PREFERRED_SPEED:
		return 0
	elif s < PREFERRED_SPEED - SPEED_THRESHOLD:
		return 0.8 						# think of it as % of "full gas"
	else:
		return 0.4 						# tweak this if the car is way over or under preferred speed
```

```python
# now Little demo to drive straight
# close to a desired speed
# - press l to exit, you need to run the bit above to start the car
cv2.namedWindow('RGB Camera', CV2.WINDOW_AUTOSIZE)
cv2.imshow('RGB Camera', camera_data['image'])

# main Loop
quit = False

while True:
	# Carla Tick
	world.tick()
	if cv2.waitKey(1) == ord('q'):
		quit = True
		break
	image = camera_data['image']
  
	steering_angle = 0 					# we do not have it yet
	# to get speed we need to use 'get velocity' function
	v = vehicle.get_velocity()
	# if velocity is a vector in 3d
	# then speed is like hypothenuse in a right triangle
	# and 3.6 is a conversion factor from meters per second to kmh
	# e.g. mh is 1000 meters and one hour is 60 min with 60 sec = 3600 sec
	speed = round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2),0)
	# now we add the speed to the window showing a camera mounted on the car
	image = cv2. putText (image, 'Speed: '+str(int(speed))+' kmh', org2,
                        font, fontScale, color, thickness, cv2. LINE_AA)
	# this is where we used the function above to determine accelerator input
	# from current speed
	estimated_throttle = maintain_speed (speed)
	# now we apply accelerator
	vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle,
                                             steer=steering_angle))
  cv2. imshow('RGB Camera', image)

#clean up 
cv2.destroyAllWindows()
camera.stop()
for actor in world.get_actors().filter('*vehicle*'):
  actor.destroy()
for sensor in world.get_actors().filter ('*sensor*'):
  sensor.destroy()
```

&nbsp;

### 转向角度

1. 汽车当前位置
2. 汽车想去的位置
3. 汽车当前角度

```python
# test steering function
world = client.get_world()
spawn_points = world.get_map().get_spawn_points()
# look for a blueprint of Mini car
vehicle_bp = world.get_blueprint_library().filter('*mini*')

start_point = spawn_points[0]
vehicle = world.try_spawn_actor(vehicle_bp[0], start_point)
# setting RGB Camera - this follow the approach explained in a Carla video
# link: https://www.youtube.com/watch?v=om8klsBj4rc&t=1184s

# camera mount offset on the car - you can tweak these to have the car in view or not
CAMERA_POS_Z = 3 
CAMERA_POS_X = -5 

camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640') # this ratio works in CARLA 9.14 on Windows
camera_bp.set_attribute('image_size_y', '360')

camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
# this creates the camera in the sim
camera = world.spawn_actor(camera_bp,camera_init_trans,attach_to=vehicle)

def camera_callback(image,data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

image_w = camera_bp.get_attribute('image_size_x').as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()

camera_data = {'image': np.zeros((image_h,image_w,4))}
# this actually opens a live stream from the camera
camera.listen(lambda image: camera_callback(image,camera_data))

cv2.namedWindow('RGB Camera',cv2.WINDOW_AUTOSIZE)
cv2.imshow('RGB Camera',camera_data['image'])

# main loop 
quit = False
curr_wp = 5 # we will be tracking waypoints in the route and switch to next one wen we get close to current one
predicted_angle = 0
while curr_wp<len(route)-1:
    # Carla Tick
    world.tick()
    if cv2.waitKey(1) == ord('q'):
        quit = True
        vehicle.apply_control(carla.VehicleControl(throttle=0,steer=0,brake=1))
        break
    image = camera_data['image']
    
    while curr_wp<len(route) and vehicle.get_transform().location.distance(route[curr_wp][0].transform.location)<5:
        curr_wp +=1 # move to next wp if we are too close
    
    predicted_angle = get_angle(vehicle,route[curr_wp][0])
    image = cv2.putText(image, 'Steering angle: '+str(round(predicted_angle,3)), org, font, fontScale, color, thickness, cv2.LINE_AA)
    v = vehicle.get_velocity()
    speed = round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2),0)
    image = cv2.putText(image, 'Speed: '+str(int(speed)), org2, font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Next wp: '+str(curr_wp), org3, font, fontScale, color, thickness, cv2.LINE_AA)
    estimated_throttle = maintain_speed(speed)
    # extra checks on predicted angle when values close to 360 degrees are returned
    if predicted_angle<-300:
        predicted_angle = predicted_angle+360
    elif predicted_angle > 300:
        predicted_angle = predicted_angle -360
    steer_input = predicted_angle
    # and conversion of to -1 to +1
    if predicted_angle<-40:
        steer_input = -40
    elif predicted_angle>40:
        steer_input = 40
    # conversion from degrees to -1 to +1 input for apply control function
    steer_input = steer_input/75

    vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
    cv2.imshow('RGB Camera',image)
    

# clean up
cv2.destroyAllWindows()
camera.stop()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()
vehicle.apply_control(carla.VehicleControl(throttle=0,steer=0,brake=1))
```

&nbsp;

&nbsp;

## 定位汽车

```python
import carla
```

```python
client = carla.Client('localhost', 2000)

world = client.get_world()
spawn_points = world.get_map().get_spawn_points()

vehicle_bp = world.get_blueprint_library().filter('*firetruck*')
start_point = spawn_points[0]
vehicle = world.try_spawn_actor(vehicle_bp[0], start_point)
```

<!--more-->

```python
# get the car's position on the map
vehicle_pos = vehicle.get_transform()
print(vehicle_pos)
```

Transform(Location(x=-64.644844, y=24.472013, z=-0.001559), Rotation(pitch=-0.000061, yaw=0.159197, roll=0.000632))

```python
# initial spawn point is the same - just 0.6m higher off the ground
print(start_point)
```

Transform (Location (x=-64.644844, y=24.471010, z=0.600000), Rotation(pitch=0.000000, yaw=0.159198, roll=0.000000))

```python
# send vehicle off
vehicle.set_autopilot(True)
```

```python
# get actual position from the car moving
vehicle_pos = vehicle.get_transform()
print(vehicle_pos)
```

Transform (Location(x=-114.478943, y=65.782814, z=-0.003669), Rotation(pitch=0.000997, yaw=90.641518, roll=0.000133))

```python
# now look at the map
town_map = world.get_map()
```

```
type (town_map)
```

Output: carla.libcarla.Map

&nbsp;

&nbsp;

