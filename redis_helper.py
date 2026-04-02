import redis
import struct
import numpy as np
import cv2

class RedisHelper:
    def __init__(self):
        self.r = redis.Redis(host='localhost', port=6379, db=0)
        self.p = self.r.pubsub()
        self.p.subscribe('frame','control')
        self.is_sim = True
        
    def convert_to_frame(self,frame_data):
        sim = self.is_sim
        if sim:
            # Convert the bytes back to a numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            # Decode the image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        h, w = struct.unpack('>II',frame_data[:8])
        frame = np.frombuffer(frame_data, dtype=np.uint8, offset=8).reshape(h,w,3).copy()
        return frame

    def from_redis(self, n):
        """Retrieve Numpy array from Redis key 'n'"""
        encoded = self.r.get(n)
        if encoded is None:
            return None
        frame = self.convert_to_frame(encoded)
        return frame
    
    def from_redis_2(self, name):
        """Retrieve and decode a frame from Redis."""
        encoded = self.r.get(name)
        if encoded is None:
            return None
        
        # Unpack the shape of the frame
        shape = struct.unpack('>II', encoded[:8])
        h, w = shape
        
        # Decode the frame data
        frame_data = encoded[8:]
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, 3))
        
        return frame

    def text_from_redis(self,n):
        encoded = self.r.get(n)
        return encoded

    def toRedis(self, name, frame):
        """Store given Numpy array 'frame' in Redis under key 'name'"""
        h, w = frame.shape[:2]
        shape = struct.pack('>II',h,w)
        encoded = shape + frame.tobytes()

        # Store encoded data in Redis
        #self.r.publish(name,encoded)

        # Use when loop usage needed
        self.r.set(name,encoded, px=500)
        # self.r.publish(name,encoded)
        