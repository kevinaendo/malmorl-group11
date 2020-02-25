from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import os
import sys
import time
import gym

sys.path.append("..")
import MalmoPython

path = os.path.abspath(MalmoPython.__file__)
print(path)

import mcenv

env = gym.make('MinecraftEnv-v0')
env.init(client_pool=[("localhost", 10000)], start_minecraft=False, allowDiscreteMovement=["move", "turn"], videoResolution=False)
# env.configure(allowDiscreteMovement=["move", "turn"], log_level="INFO")

for _ in range(10):
    t = time.time()
    env.reset()
    t2 = time.time()
    print("Startup time:", t2 - t)
    done = False
    s = 0
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        #print "obs:", obs.shape
        #print "reward:", reward
        #print "done:", done
        #print "info", info
        s += 1
    t3 = time.time()
    print((t3 - t2), "seconds total,", s, "steps total,", s / (t3 - t2), "steps/second")

env.close()
		
"""
if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)


missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Hello world!</Summary>
              </About>
              
            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FileWorldGenerator src="/Users/richardhsu/Library/MalmoPlatform/Minecraft/run/saves/maze"/>
                  <ServerQuitFromTimeUp timeLimitMs="30000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                    <Placement x="205.300" y="70.00000" z="201.700" yaw="-90"/>
                    <Inventory>
                        <InventoryItem slot="0" type="diamond_pickaxe"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <InventoryCommands/>
                  <AgentQuitFromReachingPosition>
                    <Marker x="213.700" y="70.00000" z="197.300" tolerance="0.5" description="Goal_found"/>
                  </AgentQuitFromReachingPosition>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')

# ADD YOUR CODE HERE
# TO GET YOUR AGENT TO THE DIAMOND BLOCK
agent_host.sendCommand("move 1")
agent_host.sendCommand("strafe -0.25")

# Loop until mission ends:
while world_state.is_mission_running:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission ended")
# Mission has ended.
"""
