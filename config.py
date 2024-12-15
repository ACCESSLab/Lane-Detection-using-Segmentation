
param = dict(
    model_path = "models/lane_seg_model_torchjit_serialized.pth",  ## provide the path
    image_size = (640,384),
    inference_threshold = 0.3,
    target_device = "cuda", #"cpu"
    test_image_dir = "test_images/", # provide your own directory with images
    # test_video_dir = "test_videos/", # provide your own directory with videos
    result_save_dir = "results/",
)
