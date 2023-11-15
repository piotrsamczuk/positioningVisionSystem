#include <iostream>
#include <stdio.h>
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#define LEFTCAMINDEX 0
#define RIGHTCAMINDEX 1
#define BOARDWIDTH 11
#define BOARDHEIGHT 8
#define CHECKERSIZE 12 // [mm]
#define SQUARESIZE ((CHECKERSIZE * CHECKERSIZE) * 0.01) // [sqmm]


void getCameraIndexes()
{
    for (int cameraIndex = 0; cameraIndex < 10; ++cameraIndex) {
        cv::VideoCapture cap(cameraIndex);

        // Check if the camera opened successfully
        if (cap.isOpened()) {
            std::cout << "Camera found at index: " << cameraIndex << std::endl;

            // Release the VideoCapture
            cap.release();
        }
    }
}

std::pair <cv::Mat, cv::Mat> getFramesFromCaptures(cv::VideoCapture& capLeft, cv::VideoCapture& capRight)
{
    // Read a frame from the camera
    cv::Mat frameLeft;
    cv::Mat frameRight;
    capLeft >> frameLeft;
    capRight >> frameRight;     
    // Check if the frame is empty
    if (frameLeft.empty())
    {
        std::cerr << "Error reading left frame from camera." << std::endl;
    }
    if (frameRight.empty())
    {
        std::cerr << "Error reading right frame from camera." << std::endl;
    }
    return std::make_pair(frameLeft, frameRight);
}

void viewCombinedCameraFeeds(cv::VideoCapture& capLeft, cv::VideoCapture& capRight)
{
    // Create a window to display the camera feed
    cv::namedWindow("Dual Webcams", cv::WINDOW_FREERATIO);
    while (true)
    {
        auto [frameLeft, frameRight] = getFramesFromCaptures(capLeft, capRight);
        // Flip frames vertically
        cv::flip(frameLeft, frameLeft, -1);
        cv::flip(frameRight, frameRight, -1);
        // Combine frames into one
        cv::Mat combinedFrame;
        cv::hconcat(frameLeft, frameRight, combinedFrame);
        // Display the combined frame
        cv::imshow("Dual Webcams", combinedFrame);
        // To exit press esc
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }
    cv::destroyAllWindows();
}

std::pair<cv::VideoCapture, cv::VideoCapture> startVideoCaptures(const unsigned int& leftCameraIndex, const unsigned int& rightCameraIndex)
{
    cv::VideoCapture capLeft(leftCameraIndex);
    cv::VideoCapture capRight(rightCameraIndex);
    if (not capLeft.isOpened())
    {
        std::cerr << "Error opening the left camera." << std::endl;
    }
    if (not capRight.isOpened())
    {
        std::cerr << "Error opening the right camera." << std::endl;
    }
    return std::make_pair(capLeft, capRight);
}

void drawAxis(cv::Mat &_image, cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs, cv::InputArray _rvec, cv::InputArray _tvec, float length)
{
    // project axis points
    std::vector<cv::Point3f> axisPoints;
    axisPoints.push_back(cv::Point3f(0, 0, 0));
    axisPoints.push_back(cv::Point3f(length, 0, 0));
    axisPoints.push_back(cv::Point3f(0, length, 0));
    axisPoints.push_back(cv::Point3f(0, 0, length));
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(axisPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

    // draw axis lines
    cv::line(_image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3);
    cv::line(_image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3);
    cv::line(_image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
}

void writeCalibrationData(const cv::Mat& K, const cv::Mat& D)
{
    //Writing the data in a YAML file
    std::string out_file = "calibration.txt";
    cv::FileStorage fs(out_file, cv::FileStorage::WRITE);
    fs << "K" << K;
    fs << "D" << D;
    fs << "board_width" << BOARDWIDTH;
    fs << "board_height" << BOARDHEIGHT;
    fs << "square_size" << SQUARESIZE;
    printf("Done Calibration\n");
}

void captureImagesForCalibration(cv::VideoCapture& capLeft, cv::VideoCapture& capRight, cv::Size boardSize,
    std::vector<std::vector<cv::Point2f>>& imagePoints, std::vector<std::vector<cv::Point3f>>& objectPoints, cv::Mat& matImg)
{
    // Gather user input
    int numberOfImages = 50;
    // Loop through video stream until we capture enough images (num_imgs) of checkerboard
    while(true)
    {
        capRight >> matImg;
        if(matImg.empty())
        {
            break;
        }
        // Сheckerboard corner coordinates in the image
        std::vector<cv::Point2f> corners;
        // Here we find all the corner points of each image and their corresponding 3D world points
        // and prepare the corresponding vectors.
        // Find all the checkerboard corners
        bool found = cv::findChessboardCorners(matImg, boardSize, corners, cv::CALIB_CB_FAST_CHECK);
        if(found)
        {
            cv::Size winSize = cv::Size( 12, 12 );
            cv::Size zeroZone =cv::Size( -1, -1 );
            cv::TermCriteria criteria = cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001 );
            // Convert image to grayscale images
            cv::Mat src_gray;
            cvtColor( matImg, src_gray, cv::COLOR_BGR2GRAY );
            // Find more exact corner positions (more exact than integer pixels)
            cv::cornerSubPix( src_gray, corners, winSize, zeroZone, criteria );
            // This function helps to visualize the checkerboard corners found (optional)
            cv::drawChessboardCorners(matImg, boardSize, cv::Mat(corners), found);
            // Prepare the objectPoints and imagePoints vectors
            std::vector<cv::Point3f> obj;
            for (int i = 0; i < BOARDHEIGHT - 1; i++)
            {
                for (int j = 0; j < BOARDWIDTH - 1; j++)
                {
                    obj.push_back(cv::Point3f(j * CHECKERSIZE, i * CHECKERSIZE, 0));
                }
            }
            std::cout <<"Found corners!" << std::endl;
            imagePoints.push_back(corners);
            objectPoints.push_back(obj);
        }
        cv::imshow("Camera", matImg);
        cv::waitKey(1);
        // Esc to exit if we have enough images of checkerboard
        if (imagePoints.size() == numberOfImages)
        {
            break;
        }
    }
}

void calibrate(cv::VideoCapture& capLeft, cv::VideoCapture& capRight)
{
    //Checkerboard corner coordinates in the image
    std::vector<std::vector<cv::Point2f>> imagePoints;
    //Object points are the actual 3D coordinate of checkerboard points
    std::vector<std::vector<cv::Point3f>> objectPoints;
    // Define size of calibration board 
    cv::Size boardSize(BOARDWIDTH - 1, BOARDHEIGHT - 1);
    cv::Mat matImg;
    captureImagesForCalibration(capLeft, capRight, boardSize, imagePoints, objectPoints, matImg);
    printf("Starting Calibration\n");
    //K contains the intrinsics
    cv::Mat K;
    //D contains the distortion coefficients
    cv::Mat D;
    //The rotation and translation vectors
    std::vector<cv::Mat> rvecs, tvecs;
    //Set flag to ignore higher order distortion coefficients k4 and k5.
    int flag = cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT + cv::CALIB_FIX_ASPECT_RATIO;

    cv::calibrateCamera(objectPoints, imagePoints, matImg.size(), K, D, rvecs, tvecs, flag);
    std::cout << "DUPALOG" << std::endl;
    while(true)
    {
        capLeft >> matImg;
        if(matImg.empty())
        {
            break;
        }
        //Found chessboard corners
        std::vector<cv::Point2f> imagePoints;
        bool found = cv::findChessboardCorners(matImg, boardSize, imagePoints, cv::CALIB_CB_FAST_CHECK);
        if(found)
        {
            cv::drawChessboardCorners(matImg, boardSize, cv::Mat(imagePoints), found);

            std::vector< cv::Point3f > obj;
            for (int i = 0; i < BOARDHEIGHT - 1; i++)
                for (int j = 0; j < BOARDWIDTH - 1; j++)
                    obj.push_back(cv::Point3f((float)j * CHECKERSIZE, (float)i * CHECKERSIZE, 0));

            //SolvePnP
            cv::Mat rvec, tvec;
            cv::solvePnP(obj, imagePoints, K, D, rvec, tvec);

            drawAxis(matImg, K, D, rvec, tvec, CHECKERSIZE);
            std::cout << "Distance to chessboard: "  << cv::norm(tvec) << std::endl;
        }

        cv::imshow("Camera", matImg);
        auto c = cv::waitKey(30);

        // 27 == ESC
        // !! The window with the image displayed in it must have focus (i.e. be selected) when you press the key
        if(c == 27) {
            break;
        }
    }
    writeCalibrationData(K, D);
}

int main()
{
    // Start video capture of both cameras
    auto [capLeft, capRight] = startVideoCaptures(LEFTCAMINDEX, RIGHTCAMINDEX);

    //viewCombinedCameraFeeds(capLeft, capRight);
    calibrate(capLeft, capRight);
    // Release the VideoCapture and close the window
    capLeft.release();
    capRight.release();

    return 0;
}
