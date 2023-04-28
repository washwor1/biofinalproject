from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
from PIL import Image
import pytesseract
import io
import numpy as np
import tensorflow as tf

def play_game(model):
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
        for i in (0,0,0):
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
                    loop_duration = time.time() - start_time
                    break
                

                screenshot_png = driver.get_screenshot_as_png()
                img = Image.open(io.BytesIO(screenshot_png))
                scaled_size = (220, 250)  # Adjust the scaling factor as needed
                input_img = img.resize(scaled_size, Image.Resampling.LANCZOS)
                input_data = np.expand_dims(input_img, axis=0)
                predictions = model.predict(input_data)

                print(predictions)


                time.sleep(5)

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

            img.save("game_window_screenshot.png")

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
    except:
        driver.quit()
        fitness = play_game(model)

    if fitness == 0:
        fitness = fitness_function(score,run_time) 
    return fitness


def fitness_function(score, time_alive):
    # Normalize the score and time_alive by dividing by the maximum possible values
    normalized_score = score / 100000
    normalized_time_alive = time_alive / 36000

    # Assign weights to each component
    weight_score = 0.8
    weight_time_alive = 0.2

    # Calculate the fitness value
    fitness = weight_score * normalized_score + weight_time_alive * normalized_time_alive

    return fitness
