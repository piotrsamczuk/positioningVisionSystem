#include <iostream>
#include <stdio.h>
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#define LEFTCAMINDEX 0
#define RIGHTCAMINDEX 1
#define BOARDWIDTH 11
#define BOARDHEIGHT 8
#define CHECKERSIZE 17 // [mm]
#define SQUARESIZE (CHECKERSIZE * CHECKERSIZE) // [sqmm]
#define NUMBEROFIMAGES 20
#define CUBESQUARESIZE 15 //[mm]

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

void writeCalibrationData(const cv::Mat& KL, const cv::Mat& DL, const cv::Mat& KR, const cv::Mat& DR)
{
    //Writing the data in a YAML file
    std::string outputFile = "calibration.txt";
    cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);
    if(not fs.isOpened())
    {
        std::cerr << "Error saving calibration data." << std::endl;
        return;
    }
    fs << "KL" << KL;
    fs << "DL" << DL;
    fs << "KR" << KR;
    fs << "DR" << DR;
    fs << "board_width" << BOARDWIDTH;
    fs << "board_height" << BOARDHEIGHT;
    fs << "square_size" << CHECKERSIZE;
    printf("Done writing calibration data\n");
}

void readCalibrationData()
{
    cv::Mat K, D;
    std::string inputFile = "calibration.txt";
    cv::FileStorage fs(inputFile, cv::FileStorage::READ);
    if(not fs.isOpened())
    {
        std::cerr << "Error reading calibration data." << std::endl;
        return;
    }


    printf("Done reading calibration data\n");
}

void captureImagesForCalibration(cv::VideoCapture& capLeft, cv::VideoCapture& capRight, cv::Size boardSize,
    std::vector<std::vector<cv::Point2f>>& imagePointsL, std::vector<std::vector<cv::Point2f>>& imagePointsR, std::vector<std::vector<cv::Point3f>>& objectPoints, cv::Mat& matImgL, cv::Mat& matImgR)
{
    // Loop through video stream until we capture enough images (num_imgs) of checkerboard
    while(true)
    {
        capLeft >> matImgL;
        capRight >> matImgR;
        if(matImgL.empty() or matImgR.empty())
        {
            break;
        }
        // Сheckerboard corner coordinates in the image
        std::vector<cv::Point2f> cornersL;
        std::vector<cv::Point2f> cornersR;
        // Here we find all the corner points of each image and their corresponding 3D world points
        // and prepare the corresponding vectors.
        // Find all the checkerboard corners
        bool foundL = cv::findChessboardCorners(matImgL, boardSize, cornersL, cv::CALIB_CB_ADAPTIVE_THRESH);
        bool foundR = cv::findChessboardCorners(matImgR, boardSize, cornersR, cv::CALIB_CB_ADAPTIVE_THRESH);
        cv::Size winSize = cv::Size( 12, 12 );
        cv::Size zeroZone = cv::Size( -1, -1 );
        cv::TermCriteria criteria = cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1 );
        if(foundL and foundR)
        {
            // Convert image to grayscale images
            cv::Mat grayL;
            cvtColor(matImgL, grayL, cv::COLOR_BGR2GRAY);
            // Find more exact corner positions (more exact than integer pixels)
            cv::cornerSubPix( grayL, cornersL, winSize, zeroZone, criteria );
            // This function helps to visualize the checkerboard corners found (optional)
            cv::drawChessboardCorners(matImgL, boardSize, cv::Mat(cornersL), foundL);

            // Convert image to grayscale images
            cv::Mat grayR;
            cvtColor(matImgR, grayR, cv::COLOR_BGR2GRAY);
            // Find more exact corner positions (more exact than integer pixels)
            cv::cornerSubPix( grayR, cornersR, winSize, zeroZone, criteria );
            // This function helps to visualize the checkerboard corners found (optional)
            cv::drawChessboardCorners(matImgR, boardSize, cv::Mat(cornersR), foundR);

            // Prepare the objectPoints and imagePoints vectors
            std::vector<cv::Point3f> obj;
            for (int i = 0; i < BOARDHEIGHT - 1; i++)
            {
                for (int j = 0; j < BOARDWIDTH - 1; j++)
                {
                    obj.push_back(cv::Point3f((float)j * CHECKERSIZE, (float)i * CHECKERSIZE, 0));
                }
            }
            std::cout <<"Found corners!" << std::endl;
            imagePointsL.push_back(cornersL);
            imagePointsR.push_back(cornersR);
            objectPoints.push_back(obj);
        }
        cv::imshow("Camera left", matImgL);
        cv::imshow("Camera right", matImgR);
        cv::waitKey(30);
        if (imagePointsL.size() == NUMBEROFIMAGES and imagePointsR.size() == NUMBEROFIMAGES)
        {
            break;
        }
    }
}

void calibrate(cv::VideoCapture& capLeft, cv::VideoCapture& capRight)
{
    //Checkerboard corner coordinates in the image
    //Object points are the actual 3D coordinate of checkerboard points
    std::vector<std::vector<cv::Point2f>> imagePointsL, imagePointsR;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    cv::Size boardSize(BOARDWIDTH - 1, BOARDHEIGHT - 1);
    cv::Mat matImgL, matImgR;
    captureImagesForCalibration(capLeft, capRight, boardSize, imagePointsL, imagePointsR, objectPoints, matImgL, matImgR);
    printf("Starting Calibration\n");
    //K contains the intrinsics
    //D contains the distortion coefficients
    //The rotation and translation vectors
    cv::Mat KL, DL;
    cv::Mat KR, DR;
    std::vector<cv::Mat> rvecsL, tvecsL;
    std::vector<cv::Mat> rvecsR, tvecsR;
    int flag = cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT + cv::CALIB_FIX_ASPECT_RATIO;
    double reprojectionErrorL = cv::calibrateCamera(objectPoints, imagePointsL, matImgL.size(), KL, DL, rvecsL, tvecsL, flag);
    double reprojectionErrorR = cv::calibrateCamera(objectPoints, imagePointsR, matImgR.size(), KR, DR, rvecsR, tvecsR, flag);
    std::cout << "reprojectionErrorL: " << reprojectionErrorL << std::endl;
    std::cout << "reprojectionErrorR: " << reprojectionErrorR << std::endl;

    //ponizej nie dziala

    // Essential Matrix and fundamental matrix
    // cv::Mat ES, FS;
    //The rotation and translation vectors for cameras in stereo
    // std::vector<cv::Mat> rvecsS, tvecsS;
    // double reprojectionErrorLR = cv::stereoCalibrate(objectPoints, imagePointsL, imagePointsR, matImgL, DL, matImgR, DR, matImgL.size(), rvecsS, tvecsS, ES, FS);
    // std::cout << "reprojectionErrorLR: " << reprojectionErrorLR << std::endl;
    while(true)
    {
        capLeft >> matImgL;
        capRight >> matImgR;
        if(matImgL.empty() or matImgR.empty())
        {
            std::cerr << "Error reading from camera capture after calibration." << std::endl;
            break;
        }
        //Found chessboard corners
        std::vector<cv::Point2f> imagePointsL, imagePointsR;
        bool foundL = cv::findChessboardCorners(matImgL, boardSize, imagePointsL, cv::CALIB_CB_ADAPTIVE_THRESH);
        bool foundR = cv::findChessboardCorners(matImgR, boardSize, imagePointsR, cv::CALIB_CB_ADAPTIVE_THRESH);
        if(foundL or foundR)
        {
            cv::drawChessboardCorners(matImgL, boardSize, cv::Mat(imagePointsL), foundL);
            cv::drawChessboardCorners(matImgR, boardSize, cv::Mat(imagePointsR), foundR);

            std::vector<cv::Point3f> obj;
            for (int i = 0; i < BOARDHEIGHT - 1; i++)
            {
                for (int j = 0; j < BOARDWIDTH - 1; j++)
                {
                    obj.push_back(cv::Point3f((float)j * CHECKERSIZE, (float)i * CHECKERSIZE, 0));
                }
            }
            //SolvePnP
            cv::Mat rvecL, tvecL;
            cv::Mat rvecR, tvecR;
            cv::solvePnP(obj, imagePointsL, KL, DL, rvecL, tvecL);
            cv::solvePnP(obj, imagePointsR, KR, DR, rvecR, tvecR);

            drawAxis(matImgL, KL, DL, rvecL, tvecL, CHECKERSIZE);
            drawAxis(matImgR, KR, DR, rvecR, tvecR, CHECKERSIZE);
            std::cout << "Distance to chessboard from left camera: "  << cv::norm(tvecL) << std::endl;
            std::cout << "Distance to chessboard from right camera: "  << cv::norm(tvecR) << std::endl;
        }

        cv::imshow("Camera left", matImgL);
        cv::imshow("Camera right", matImgR);
        auto c = cv::waitKey(30);

        // 27 == ESC
        // !! The window with the image displayed in it must have focus (i.e. be selected) when you press the key
        if(c == 27) {
            break;
        }
    }
    writeCalibrationData(KL, DL, KR, DR);
}

int main()
{
    // Start video capture of both cameras
    auto [capLeft, capRight] = startVideoCaptures(LEFTCAMINDEX, RIGHTCAMINDEX);

    // viewCombinedCameraFeeds(capLeft, capRight);
    calibrate(capLeft, capRight);

    // Release the VideoCapture and close the window
    capLeft.release();
    capRight.release();

    return 0;
}
