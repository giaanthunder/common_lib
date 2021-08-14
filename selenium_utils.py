import os, sys, time, math
import subprocess as sp

from selenium import webdriver
from selenium.webdriver.firefox.options import Options as ffOptions
from selenium.webdriver.chrome.options import Options as chrOptions
from selenium.webdriver.opera.options import Options as opeOptions
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.opera import OperaDriverManager
from selenium.webdriver.common.keys import Keys

# sys.path.append('/media/anhuynh/DATA/03_task/common_lib')

def wait_for(driver, xpath, dbg=False):
    while True:
        try:
            if dbg:
                print('Wait for', xpath)
            element = driver.find_element_by_xpath(xpath)
            time.sleep(1)
            break
        except:
            time.sleep(1)
    return element

def exists(driver, xpath):
    try:
        print('Check exist', xpath)
        element = driver.find_element_by_xpath(xpath)
        return True
    except:
        return False

def find(driver, xpath):
    try:
        element = driver.find_element_by_xpath(xpath)
        return element
    except:
        return None

def finds(driver, xpath):
    try:
        elements = driver.find_elements_by_xpath(xpath)
        return elements
    except:
        return []

def wait_in_second(driver, xpath, second, dbg=False):
    element = None
    for i in range(second):
        try:
            if dbg:
                print('Wait in second', xpath)
            element = driver.find_element_by_xpath(xpath)
            time.sleep(1)
            break
        except:
            time.sleep(1)
    return element


def try_click_element(driver, xpath, delay=1):
    try:
        print('Try click', xpath)
        element = driver.find_element_by_xpath(xpath)
        element.click()
        time.sleep(delay)
        return True
    except:
        return False

def try_input_element(driver, xpath, value, delay=1):
    try:
        print('Try input', xpath)
        element = driver.find_element_by_xpath(xpath)
        element.clear()
        element.send_keys(value)
        time.sleep(delay)
        return True
    except:
        return False

def login(driver, url_info, name_info, pw_info):
    url, login_xp = url_info
    name, name_xp = name_info
    pw, pw_xp     = pw_info

    driver.get(url)
    wait_for(driver, name_xp)
    try_input_element(driver, name_xp, name)
    try_input_element(driver, pw_xp, pw)
    return try_click_element(driver, login_xp)

def download(url, path):
    cmd = "python download.py '%s' '%s' &"%(url,path)
    sp.call(cmd, shell=True)

def download2(url, path):
    cmd = "wget '%s' -O '%s' &"%(url,path)
    sp.call(cmd, shell=True)

def chrome(opt=[], ext=[]):
    CHROMEDRIVER_PATH = ChromeDriverManager().install()
    options = chrOptions()
    for o in opt:
        options.add_argument("--%s"%o) # ['disable-notifications', 'disable-infobars', 'mute-audio']
    for e in ext:
        options.add_extension(e)
    driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, options=options)
    return driver

def firefox(opt=[], ext=[]):
    FIREFOXDRIVER_PATH = GeckoDriverManager().install()
    options = ffOptions()
    for o in opt:
        options.add_argument("--%s"%o)
    for e in ext:
        options.add_extension(e)
    driver = webdriver.Firefox(executable_path=FIREFOXDRIVER_PATH, options=options)
    return driver

def open_tab(driver):
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[-1])

def close_tab(driver):
    driver.close()
    driver.switch_to.window(driver.window_handles[-1])

def http_req(driver, req_type, request):
    cmd = '''
        var xhttp = new XMLHttpRequest();
        xhttp.open("%s", "%s", true);
        xhttp.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                var element = document.createElement("P");
                element.id = "this_is_a_unique_name"
                element.innerText = this.response;
                element.style.display = "none";
                document.body.insertBefore(element, document.body.firstChild);
                // document.body.appendChild(element);
            }
        };
        xhttp.send();
    '''%(req_type,request)
    driver.execute_script(cmd)
    response = wait_in_second(driver, "//p[@id='this_is_a_unique_name']", 10,False)

    if response is not None:
        response = response.get_attribute('innerText')
        cmd = '''
            var element = document.querySelector("#this_is_a_unique_name");
            element.remove();
        '''
        driver.execute_script(cmd)
    else:
        response = ''
    return response

def get_scroll_height(driver):
    return driver.execute_script("return document.body.scrollHeight")

def send_key(driver, key):
    body = driver.find_element_by_tag_name('body')
    if key == 'END':
        body.send_keys(Keys.END)

    # ADD ALT ARROW_DOWN ARROW_LEFT ARROW_RIGHT ARROW_UP BACKSPACE BACK_SPACE 
    # CANCEL CLEAR COMMAND CONTROL DECIMAL DELETE DIVIDE DOWN END ENTER EQUALS 
    # ESCAPE F1 (1->12) HELP HOME INSERT LEFT LEFT_ALT LEFT_CONTROL LEFT_SHIFT 
    # META MULTIPLY NULL NUMPAD0 (0->9) PAGE_DOWN PAGE_UP PAUSE RETURN RIGHT 
    # SEMICOLON SEPARATOR SHIFT SPACE SUBTRACT TAB UP
    # Keys.CONTROL + 't'

