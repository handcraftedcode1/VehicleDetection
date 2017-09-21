from moviepy.video.io.VideoFileClip import VideoFileClip
from detection_pipeline import DetectionPipeline

short = "videos/short_movie.mp4"
test = "videos/test_video.mp4"
project = "videos/project_video.mp4"
challenge = "videos/challenge_video.mp4"

input_video = challenge
debug = False  # output a debugging version of the video

if __name__ == '__main__':
    """
    Process each frame of a video to detect lanes and render output video
    """
    video_clip = VideoFileClip(input_video)

    pipeline = DetectionPipeline()

    if debug:
        processed_clip = video_clip.fl_image(pipeline.debug_frame)
    else:
        processed_clip = video_clip.fl_image(pipeline.process_frame)

    # save video
    processed_clip.write_videofile(input_video[:-4] + '_processed.mp4', audio=False)

    print('Done')
