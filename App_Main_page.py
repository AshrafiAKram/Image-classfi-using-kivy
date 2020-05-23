import os
import cv2
import kivy
import numpy as np
from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.uix.image import Image
import dlib
from DB_prcess import dis_classfi
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.graphics.texture import Texture
from kivy.graphics import Color
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.core.window import Window

Window.clearcolor = (0.5,0.5,0.5,1)


detect = dlib.get_frontal_face_detector()
pradict = dlib.shape_predictor( os.path.join(os.getcwd(),'shape_predictor_68_face_landmarks.dat'))

# global variable
local_img = []


def cv2_texture(img):
    buf1 = cv2.flip(img, 0)
    buf = buf1.tostring()
    texture1 = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
    texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return texture1


image_path =[]


class BrowserWidow(Screen):
    def load(self,filename):
        path = filename[0]
        image_path.append(path)

class MainWindow(Screen):
    pass_frame = ObjectProperty(None)

    def but_image_view(self):
        self.img = cv2.imread(image_path[-1])

        texture = cv2_texture(self.img)
        self.pass_frame.texture = texture


    def build(self):
        self.img1=Image()
        layout = Widget()
        layout.add_widget(self.img1)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, dt):
        try:
            ret, frame = self.capture.read()
            self.img = frame
            texture1 = cv2_texture(self.img)
            self.pass_frame.texture = texture1
        except:
            pass

    def pass_face(self):
        local_img.append(self.img)

    def browser_show(self):
        Control_browser().show_popup()


class Result_window(Screen):
    classfi_name = ObjectProperty(None)

    def image_classfi_fun(self):
        self.classfi_name.text = ''
        dis_name_value = dis_classfi(local_img)
        self.classfi_name.text = dis_name_value



class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("my.kv")

class MyMobile(App):
    def build(self):
        return kv

class Control_browser:

    def show_popup(self):
        show= BrowserWidow()
        self.popupwindo= Popup(title = 'Browser', content=show, size_hint=(None, None), size=(400,400))
        self.popupwindo.open()


if __name__ == '__main__':
    MyMobile().run()