import cv2
import enum

### GUI ###
def text_with_backgroud(img, text, org, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, scale=1, color_text=(0,0,0), color_bg=(255,255,255)):
    (txt_w, txt_h) = cv2.getTextSize(text, font, fontScale=scale, thickness=1)[0]
    cv2.rectangle(img, tuple([org[0], org[1]-txt_h-3]), tuple([org[0]+txt_w, org[1]]), color_bg, cv2.FILLED)
    cv2.putText(img, text, tuple([org[0],org[1]-3]), font, scale, color_text)

### Enum ###
class RobotFlag(enum.Enum):
    Work = 1
    Idle = 2
    Warn = 3
    Dead = 4
