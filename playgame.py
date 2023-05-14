from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import io
from PIL import Image
import pytesseract
import io
import numpy as np
import tensorflow as tf

def play_game(model, fail = 0):

    fail_count = 0
    fail_count = fail
    center = (500,440)
    loop_duration = 0
    mouse = [0,0]
    score = 0
    run_time = 0
    fitness = 0.0
    fps_counter_js = """
    (function() {
        let counter = document.createElement('div');
        counter.style.position = 'fixed';
        counter.style.top = '10px';
        counter.style.left = '500px';
        counter.style.padding = '5px 10px';
        
        counter.style.color = 'white';
        counter.style.fontFamily = 'monospace';
        counter.style.zIndex = 9999;
        document.body.appendChild(counter);

        let lastFrameTime = performance.now();
        let frameCount = 0;

        function updateCounter() {
            frameCount++;
            let currentTime = performance.now();
            let elapsedTime = currentTime - lastFrameTime;

            if (elapsedTime >= 1000) {
                let fps = (frameCount / elapsedTime) * 1000;
                counter.textContent = fps.toFixed(1) + ' FPS';
                frameCount = 0;
                lastFrameTime = currentTime;
            }

            requestAnimationFrame(updateCounter);
        }

        updateCounter();
    })();
    """



    chrome_options = Options()
    chrome_options.add_argument('--window-size=1000,1000')
    chrome_options.add_argument("--gpu-driver")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get('https://diep.io/')
    action_chains = ActionChains(driver)


    while True:
        screen_element = driver.execute_script('return document.querySelector("d-base.diep-native").shadowRoot.querySelector("d-home");')
        if screen_element:
            break
        time.sleep(1)

    print("Found")

    shadow_root = driver.execute_script("return arguments[0].shadowRoot", screen_element)

    element = WebDriverWait(shadow_root, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "d-menu"))
        )
    #menu-grid > div:nth-child(1) > d-button
    innersr = driver.execute_script("return arguments[0].shadowRoot", element)

    element = WebDriverWait(innersr, 1).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#menu-grid > div:nth-child(1) > d-button"))
        )
    element.click()  


    element = WebDriverWait(innersr, 1).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#submenu-grid > d-button:nth-child(5)"))
        )
    element.click() 
    driver.execute_script(fps_counter_js)

    try:
        mouse = moveMouseTo(action_chains, mouse, center)
        for i in (1,2,3):
            image_list = []
            start_time = time.time()
            time.sleep(2)
            while True:
                screen_element = driver.execute_script('return document.querySelector("d-base.diep-native").shadowRoot.querySelector("d-home");')
                if screen_element:
                    break
                time.sleep(1)

            shadow_root = driver.execute_script("return arguments[0].shadowRoot", screen_element)

            element = WebDriverWait(shadow_root, 1).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#username-row > d-button"))
                )
            element.click()   
            
            while True:
                screen_element = driver.execute_script('return document.querySelector("d-base.diep-native").shadowRoot.querySelector("d-stats");')
                if screen_element:
                    # width, height = image_list[0].size
                    # loop_duration = time.time() - start_time
                    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    # video = cv2.VideoWriter(f'recording{i}.avi', fourcc, len(image_list)/loop_duration, (width, height))
                    # for image in image_list:
                    #     # Convert the PIL image to an OpenCV compatible format
                    #     open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    #     video.write(open_cv_image)
                    # # Release the video file
                    # video.release()
                    break
                

                
                #take screenshot, open it with PIL, add it to recording list
                screenshot_png = driver.get_screenshot_as_png()
                img = Image.open(io.BytesIO(screenshot_png))
                # image_list.append(img)

                # #Convert image to numpy array
                img_arr = np.array(img)
                img_arr = np.expand_dims(img_arr, axis=0)

                #made a prediction using screenshot
                predictions = model.predict(img_arr)
                
                #take output tensor from the model and send the controls to the browser using action chains.
                mouse = handleInputs(predictions, action_chains, mouse)


            time.sleep(2)
            screenshot_png = driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(screenshot_png))
            score_area = (420,385,520,412)
            score_img = img.crop(score_area)
            scaled_size = (100 * 3, 27 * 3)  # Adjust the scaling factor as needed
            score_img = score_img.resize(scaled_size, Image.Resampling.LANCZOS)
            score_img = score_img.convert('L')
            threshold = 195
            score_img = score_img.point(lambda p: p > threshold and 255)

            score_img.save("game_window_screenshot.png")

            score_text = pytesseract.image_to_string(score_img, config="--psm 6 --oem 3")
            score_text = score_text.replace(',', '')
            score_text = score_text.replace(')', '')
            score_text = score_text.replace('.', '')
            if (score_text == '\n'):
                score_text = '0'
            score += int(score_text)
            run_time+=loop_duration
            print(int(score_text))
            time.sleep(2)
            shadow_root = driver.execute_script("return arguments[0].shadowRoot", screen_element)
            element = WebDriverWait(shadow_root, 1).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#continue"))
                )
            element.click() 
    except Exception as e:
        print(e)
        driver.quit()
        fail_count += 1
        fitness, fail_count = play_game(model,fail_count)

    if fitness == 0:
        fitness = fitness_function(score,run_time) 
    return fitness, fail_count


def fitness_function(score, time_alive):
    # Normalize the score and time_alive by dividing by the maximum possible values
    normalized_score = score / 50000
    normalized_time_alive = time_alive / 6000

    # Assign weights to each component
    weight_score = 0.8
    weight_time_alive = 0.2

    # Calculate the fitness value
    fitness = weight_score * normalized_score + weight_time_alive * normalized_time_alive

    return fitness




def moveMouseTo(action, mouse, pos):
    x_offset = pos[0] - mouse[0]
    y_offset = pos[1] - mouse[1]
    action.move_by_offset(x_offset, y_offset).perform()
    mouse[0] += x_offset
    mouse[1] += y_offset
    return mouse


def handleInputs(prediction, action, mouse):
    if (prediction[0][0] > 0):
        action.key_down('w')
    else:
        action.key_up('w')
    if (prediction[0][1] > 0):
        action.key_down('a')
    else:
        action.key_up('a')
    if (prediction[0][2] > 0):
        action.key_down('s')
    else:
        action.key_up('s')
    if (prediction[0][3] > 0):
        action.key_down('d')
    else:
        action.key_up('d')
    if (prediction[0][4] > 0):
        action.send_keys('1')
    if (prediction[0][5] > 0):
        action.send_keys('2')
    if (prediction[0][6] > 0):
        action.send_keys('3')
    if (prediction[0][7] > 0):
        action.send_keys('4')
    if (prediction[0][8] > 0):
        action.send_keys('5')
    if (prediction[0][9] > 0):
        action.send_keys('6')
    if (prediction[0][10] > 0):
        action.send_keys('7')
    if (prediction[0][11] > 0):
        action.send_keys('8')
    if (prediction[0][12] > 0):
        action.click_and_hold()
    else:
        action.release()
    
    offset_X = 500 + max(min(prediction[0][13] * 100, 100), -100)
    offset_Y = 440 + max(min(prediction[0][14] * 100, 100), -100)

    mouse = moveMouseTo(action, mouse, (offset_X, offset_Y))

    action.perform()
    return mouse


