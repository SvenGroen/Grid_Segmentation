import os
import time
import datetime

start_time = time.time()

while True:
    time_passed = (time.time() - start_time) / 60.
    with open(str("time_test.txt"), "w") as txt_file:
        txt_file.write("Time Passed: {} min.\n".format(time_passed))
    time.sleep(60)
