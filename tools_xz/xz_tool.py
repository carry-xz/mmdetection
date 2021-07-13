import cv2 
import time, os, shutil, json 


def get_timestr():
    temp = time.time()
    ms = int((temp-int(temp))*1000)
    return time.strftime("%Y%m%d_%H%M%S_", time.localtime())+str(ms)

def get_files(files_dir, file_type=('jpg', 'png', 'JPG', 'PNG')):
    res = []
    for root, dirs, files in os.walk(files_dir):
        for file in files:
            if file.endswith(file_type):
                res.append(os.path.join(root, file))
    return res

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data 

def save_json(data_dic, json_file):
    with open(json_file, 'w') as f:
        json.dump(data_dic, f)
        print('finish save json:',json_file) 
    
    
class Video():
    '''read and write video
    eg.
    video = Video()
    video.init_video_reader(vedio_path)
    video.init_video_writer(save_pth)
    ret, frame = video.read()
    '''
    def __init__(self) -> None:
        self.video_reader_has_init = False
        self.video_writer_has_init = False
        self.count = 0

    def init_video_reader(self, vid_path):
        self.vid_pth = vid_path
        self.vid_reader = cv2.VideoCapture(vid_path)
        self.width = self.vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        self.height = self.vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        self.fps = self.vid_reader.get(cv2.CAP_PROP_FPS)
        self.video_reader_has_init = True 
        print(self.info())

    def init_video_writer(self, save_path,fps=None,width=None,height=None):
        if fps is None and self.video_reader_has_init:
            fps = self.fps
            width = self.width
            height = self.height
        self.vid_pth = save_path
        self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
        self.video_writer_has_init = True
        print('save pth:', save_path)

    def info(self):
        msg = "video path:{}\nfps:{}\nwidth:{}\nheight:{}".format(self.vid_pth,
        self.fps,self.width,self.height)
        return msg

    def write(self, frame):
        assert self.video_writer_has_init==True
        self.count += 1
        self.cost = time.time() - self.start_
        self.vid_writer.write(frame)
        
    def read(self):
        assert self.video_reader_has_init==True
        retval, frame = self.vid_reader.read()
        self.start_ = time.time()
        return retval, frame

    def close_reader(self):
        self.vid_reader.release()

    def close_writer(self):
        self.vid_writer.release()

    def close(self):
        self.close_writer()
        self.close_reader()