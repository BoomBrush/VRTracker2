# Standard imports
import cv2, time
import numpy as np
import zmq, json, time, sys, struct
from collections import namedtuple
from construct import Int32ub, Int32ul, Float32l, Struct, Const, Padded, Array
from math import pi

# OpenCV Webcam setting up
params = cv2.SimpleBlobDetector_Params()
params.blobColor = 255
#params.minArea = 70
#params.maxArea = xxx
#params.filterByCircularity = 1
#params.filterByInertia = 1
detector = cv2.SimpleBlobDetector_create(params)

cap = cv2.VideoCapture(0)

#cap.set(3,320)
#cap.set(4,240)
#cap.set(5,120)
cap.set(3,640)
cap.set(4,480)
cap.set(5,60)

#All this class does is prints messages to the screen so I can see what going on.
class debug: 
    def __init__(self, socket):
        self.socket = socket
        
    def send(self, text):
        self.socket.send(text)
        print("Send: " + str(text))

    def recv(self):
        recieved = self.socket.recv()
        print("Recived: " + str(recieved))
        return recieved

# This captures the image from the webcam and gets the blob, returns the position
def get_ir_position():
    before = time.time()
    ret, im = cap.read()

    keypoints = detector.detect(im)
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if len(keypoints) == 1:
        point = keypoints[0]
        return len(keypoints), point.pt[0], point.pt[1], point.size, im_with_keypoints
    else:
        return len(keypoints), 0, 0, 0, im_with_keypoints

def send_position(x, y, z, structure, endpoint):
    # position = [0.0, 0.0, 0.0, x, y, z]
    position = [x, y, z]
    byte_packet = structure.build(dict(data=position))
    endpoint.send(byte_packet)
    #print("Send: " + str(structure.parse(byte_packet)))
    #print("Recieved: " + str(structure.parse(endpoint.recv())))
    endpoint.recv()

# Setup some fancy ZMQ socket stuff and connect to the vridge api
context = zmq.Context()
control_channel = context.socket(zmq.REQ)
control_channel.connect("tcp://127.0.0.1:38219")
# First you connect to a control channel (setting up debug class)
vridge_control = debug(control_channel)

# Say hi (this doesnt do anything just confirms everything works)
vridge_control.send('{"ProtocolVersion":1,"Code":2}')
vridge_control.recv()

# Request special connection for head tracking stuff
vridge_control.send('{"RequestedEndpointName":"HeadTracking","ProtocolVersion":1,"Code":1}')
newconnection = json.loads(vridge_control.recv())
#vridge_control.close() # Close socket

# Connect to new socket (timeout is normally 15 seconds)
endpoint_address = newconnection['EndpointAddress']
endpoint = context.socket(zmq.REQ)
endpoint.connect(endpoint_address)
# Connect to the endpoint channel (setting up debug class)
vridge_endpoint = debug(endpoint)

# Specify the structure for the fancy position matrix
structure = Struct(
        Const(Int32ul, 2),
        # Const(Int32ul, 3), 
        Const(Int32ul, 5),
        Const(Int32ul, 24),
        "data" / Padded(64, Array(3, Float32l)),
)

# Capture the first blob, keep looping until you see it
while True:
    num_blobs, x, y, size, frame = get_ir_position()

    if num_blobs == 1:
        print("INITIALIZED blob at offset position X:",x,", Y:",y)
        break

    cv2.imshow("Keypoints", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

x_offset = x
y_offset = y
size_offset = size

steamvr = {'x': 0.0, 'y': 0.0, 'z': 0.0}
steamvr_offset = {'x': 0.0, 'y': 0.5, 'z': 0.0}
ppm = 250.0 # Pixels per metre

while True:
    num_blobs, x_pos, y_pos, size, frame = get_ir_position()

    if num_blobs == 1:
        x = x_pos
        y = y_pos
    else:
        print("Blobs detected: ", num_blobs)

    steamvr['x'] = 0.0
    steamvr['y'] = -(y - y_offset) / ppm + steamvr_offset['y']
    steamvr['z'] = -(x - x_offset) / ppm + steamvr_offset['x']

    # print(y, y_offset, ppm, steamvr_offset['y'])
    
    send_position(steamvr['x'], steamvr['y'], steamvr['z'], structure, endpoint)
    
    cv2.imshow("Keypoints", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
