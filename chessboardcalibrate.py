import numpy as np
import cv2 as cv

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), f"Can't open video: {video_file}"

    img_select = []
    while True:
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            key = cv.waitKey(wait_msec)
            if key == ord(' '):
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):
                    img_select.append(img)
            if key == 27:
                break

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0, "No valid chessboard detected!"

    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)

    image_size = gray.shape[::-1]
    if K is None: K = np.zeros((3, 3))
    if dist_coeff is None: dist_coeff = np.zeros(5)

    return cv.calibrateCamera(obj_points, img_points, image_size, K, dist_coeff, flags=calib_flags)

def save_undistorted_video(input_video, K, dist_coeff, output_file='chessboard_aftercalibrate.avi'):
    cap = cv.VideoCapture(input_video)
    assert cap.isOpened(), f"Cannot open video: {input_video}"

    fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    out = cv.VideoWriter(output_file, cv.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, K, (w, h), cv.CV_32FC1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        undistorted = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR)
        out.write(undistorted)

    cap.release()
    out.release()

if __name__ == '__main__':
    video_file = 'chessboard.avi'
    board_pattern = (8, 6)       
    board_cellsize = 0.025        

    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0

    rms, K, dist, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    print("\n## [1차 캘리브레이션 결과]")
    print(f"* 선택된 이미지 수: {len(img_select)}")
    print(f"* RMS 오차: {rms:.4f}")
    print(f"* 카메라 행렬 K:\n{K}")
    print(f"* 왜곡 계수: {dist.flatten()}")

    flags = cv.CALIB_USE_INTRINSIC_GUESS
    rms2, K2, dist2, rvecs, tvecs = calib_camera_from_chessboard(
        img_select, board_pattern, board_cellsize,
        K=K, dist_coeff=dist, calib_flags=flags
    )

    print("\n## [2차 캘리브레이션 결과 (Refined)]")
    print(f"* RMS 오차: {rms2:.4f}")
    print(f"* 카메라 행렬 K:\n{K2}")
    print(f"* 왜곡 계수: {dist2.flatten()}")

    save_undistorted_video(video_file, K2, dist2, output_file='chessboard_aftercalibrate.avi')
