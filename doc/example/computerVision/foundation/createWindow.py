import cv2

# 创建窗口
cv2.namedWindow('test',cv2.WINDOW_NORMAL)
# 更改窗口：window_autosize，设置大小无用
cv2.resizeWindow('test',width=800,height=600)
# 展示窗口
cv2.imshow('test',0)
# 等待按键，将窗口堵塞。会返回按键的ASCII码
key = cv2.waitKey(0)
# ord 获取字符的ASCII码
if key == ord('q'):
    # 销毁窗口
    cv2.destroyAllWindows()

