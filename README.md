# Digit Recognizer

This is a neural network trained to recognize handwritten digits. It was trained using the MNIST database and currently has a 4.4% error rate, which I am working on reducing to <1%. 

### Running the network against MNIST

You can run the network and see how it works on the MNIST test data by running `python3 run_network.py` in the shell, and can also edit the variables on the top of the script to change the epochs, batch size, number of test cases run and more

### Running the network on custom handwritten digits

You can use the network to see if it recognizes custom handwritten digits by uploading the picture of the digit (Just one digit) into the _customTestData_ folder, and then run `python3 recognize_handwritten_digits.py` in the shell. You can change the variables on top of the script to change the inputs mentioned before and also if you want the script to write a copy of the processed image into the folder _customTestDataModified_

#### Created by Chinmay Phulse