import cv2

class Tracker():
    '''Instantiates the tracker: 
        - Number of frames missed
        - Bouding box position
        - Initializes KCF tracker
    '''
    def __init__(self, img, pos):
        self.frames_disapeared = 0
        self.pos = pos # (Xi, Yi, w, h)
        self.trk_center = [int(pos[0] + pos[2]/2), int(pos[1] + pos[3]/2)]
        self.trk = cv2.TrackerKCF_create()
        self.trk.init(img, pos)

class objects_updator():
    '''Instantiates the object that will manager the trackers (Add, delete and update): 
        - List of all trackers
        - Number of frames max that a tracker can be missed (without matches)
    '''
    def __init__(self, max_disapeared = 20):
        self.trackers_list = []
        self.max_disapeared = max_disapeared

    # Verifies the matches between trackers and detections, returning new detections, trackers missed, and trackers matches
    def verify_dets(self, dets):
        matched_trk = []
        unmatched_trk = [] 
        matched_dets = []
        new_dets = []
        
        for id_trk, trk in enumerate(self.trackers_list):
            for id_det, pos in enumerate(dets):
                xi, yi, xf, yf = [pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]
                if ( trk.trk_center[0] >= xi and trk.trk_center[0] <= xf ) and ( trk.trk_center[1] >= yi and trk.trk_center[1] <= yf ):
                    matched_trk.append(id_trk)
                    matched_dets.append(id_det)
            
            if id_trk not in matched_trk:
                print("Unmatched")
                unmatched_trk.append(id_trk)

        for idx_dets in range(len(dets)):
            if idx_dets not in matched_dets:
                new_dets.append(idx_dets)

        return matched_trk, unmatched_trk, new_dets

    def update_trks(self, img):
        for idx, trk in enumerate(self.trackers_list):
            (success, trk.pos) = trk.trk.update(img)
            if success:
                (x, y, w, h) = [int(v) for v in trk.pos]
                trk.trk_center = [int(x + w/2), int(y + h/2)] # Tracker centroid
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            else:
                self.trackers_list.pop(idx)


    def update_dets(self, img, dets):
        # Returns the ID's of the matched/unmatched trackers and the id of new detections
        matched_trk, unmatched_trk, new_dets = self.verify_dets(dets) 

        for idx in new_dets:
            self.trackers_list.append(Tracker(img, dets[idx]))

        for idx in unmatched_trk:
            self.trackers_list[idx].frames_disapeared += 1
            if self.trackers_list[idx].frames_disapeared == self.max_disapeared:
                self.trackers_list.pop(idx)

        for idx in matched_trk:
            self.trackers_list[idx].frames_disapeared = 0

        return new_dets



