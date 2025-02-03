import cv2
import mediapipe as mp

dedos = [8, 12, 16, 20]
todos_dedos = [4, 8, 12, 16, 20]
dedos_aux = [4, 12, 16, 20]
dedos_g = [12, 16, 20]


def letra_a(imgem, coordenadas):
    for x in dedos:
        if coordenadas[x][1] < coordenadas[x - 2][1]:
            return False
    if coordenadas[4][0] < coordenadas[3][0]:
        return False
    if (coordenadas[8][0] - coordenadas[5][0]) > 50:
        return False
    cv2.putText(imgem, 'A', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_b(imgem, coordenadas):
    for x in dedos:
        if coordenadas[x][1] > coordenadas[x - 2][1]:
            return False
    if coordenadas[4][0] > coordenadas[2][0]:
        return False
    cv2.putText(imgem, 'B', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_c(imgem, coordenadas):
    for x in todos_dedos:
        if coordenadas[x][0] < coordenadas[x - 1][0]:
            return False
    if (coordenadas[4][1] - coordenadas[8][1]) < 30:
        return False
    if coordenadas[8][1] < coordenadas[7][1]:
        return False
    cv2.putText(imgem, 'C', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_d(imgem, coordenadas):
    if coordenadas[8][1] > coordenadas[7][1]:
        return False
    for x in dedos_aux:
        if coordenadas[x][0] < coordenadas[x - 1][0]:
            return False
    if (coordenadas[4][1] - coordenadas[12][1]) > 20:
        return False
    if coordenadas[5][0] > coordenadas[10][0]:
        return False
    cv2.putText(imgem, 'D', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_e(imgem, coordenadas):
    if coordenadas[4][0] > coordenadas[2][0]:
        return False
    for x in dedos:
        if coordenadas[x][1] < coordenadas[x - 1][1] or coordenadas[x - 2][1] > coordenadas[x - 3][1]:
            return False
        if coordenadas[x - 1][1] > coordenadas[x - 3][1]:
            return False
    cv2.putText(imgem, 'E', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_f(imgem, coordenadas):
    for x in dedos_aux:
        if coordenadas[x][1] > coordenadas[x - 2][1]:
            return False
    if coordenadas[4][1] > coordenadas[8][1]:
        return False
    if coordenadas[8][0] > coordenadas[4][0]:
        return False
    cv2.putText(imgem, 'F', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_g(imgem, coordenadas):
    if coordenadas[8][1] > coordenadas[6][1] or coordenadas[4][1] > coordenadas[2][1]:
        return False
    if coordenadas[5][0] < coordenadas[10][0] or coordenadas[4][0] < coordenadas[5][0]:
        return False
    for x in dedos_g:
        if coordenadas[x][1] < coordenadas[x - 2][1]:
            return False
    if coordenadas[4][1] > coordenadas[5][1]:
        return False
    cv2.putText(imgem, 'G', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_i(imgem, coordenadas):
    if coordenadas[20][1] > coordenadas[19][1] or coordenadas[0][1] < coordenadas[13][1]:
        return False
    if coordenadas[4][0] > coordenadas[3][0]:
        return False
    for x in todos_dedos[1:4]:
        if coordenadas[x][1] < coordenadas[x - 2][1]:
            return False
    cv2.putText(imgem, 'I', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_l(imgem, coordenadas):
    if coordenadas[4][0] < coordenadas[3][0]:
        return False
    if coordenadas[8][1] > coordenadas[7][1] or coordenadas[5][0] < coordenadas[10][0]:
        return False
    for x in todos_dedos[2:4]:
        if coordenadas[x][1] < coordenadas[x - 2][1]:
            return False
    if coordenadas[4][1] < coordenadas[5][1]:
        return False
    cv2.putText(imgem, 'L', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_m(imgem, coordenadas):
    for x in todos_dedos[2:4]:
        if coordenadas[x][1] < coordenadas[x - 1][1]:
            return False
    if coordenadas[4][0] > coordenadas[3][0] or coordenadas[0][1] > coordenadas[13][1]:
        return False
    if coordenadas[20][1] < coordenadas[19][1]:
        return False
    cv2.putText(imgem, 'M', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_n(imgem, coordenadas):
    for x in todos_dedos[1:2]:
        if coordenadas[x][1] < coordenadas[x - 1][1]:
            return False
    if coordenadas[4][0] > coordenadas[3][0] or coordenadas[0][1] > coordenadas[13][1]:
        return False
    if coordenadas[16][1] > coordenadas[15][1] or coordenadas[12][1] < coordenadas[11][1]:
        return False
    cv2.putText(imgem, 'N', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_o(imgem, coordenadas):
    for x in dedos:
        if coordenadas[x][0] < coordenadas[x - 2][0] or coordenadas[x][1] > coordenadas[2][1]:
            return False
    if coordenadas[4][1] > coordenadas[16][1] or coordenadas[0][1] < coordenadas[2][1]:
        return False
    cv2.putText(imgem, 'O', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_p(imgem, coordenadas):
    for x in dedos[0:2]:
        if coordenadas[x][0] < coordenadas[x - 1][0] or coordenadas[x][0] < coordenadas[x - 2][0]:
            return False
    if coordenadas[0][0] > coordenadas[1][0]:
        return False
    if coordenadas[6][1] > coordenadas[14][1] or coordenadas[10][1] > coordenadas[14][1] or coordenadas[13][1] < \
            coordenadas[9][1] or coordenadas[8][1] > coordenadas[12][1] or coordenadas[13][1] > coordenadas[20][1] or \
            coordenadas[13][1] > coordenadas[17][1] or coordenadas[0][1] > coordenadas[17][1]:
        return False
    cv2.putText(imgem, 'P', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_q(imgem, coordenadas):
    if coordenadas[0][1] > coordenadas[20][1] or coordenadas[0][1] > coordenadas[1][1] or coordenadas[7][1] > \
            coordenadas[8][1] or coordenadas[12][1] > coordenadas[11][1]:
        return False
    for x in dedos[2:]:
        if coordenadas[x][1] > coordenadas[x - 2][1]:
            return False
    cv2.putText(imgem, 'Q', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_r(imgem, coordenadas):
    if coordenadas[4][0] > coordenadas[3][0]:
        return False
    if coordenadas[8][1] != coordenadas[12][1] or coordenadas[8][1] < coordenadas[6][1] or coordenadas[12][1] < coordenadas[10][1]:
        return False
    if coordenadas[4][0] > coordenadas[5][0]:
        return False
    cv2.putText(imgem, 'R', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_s(imgem, coordenadas):
    for x in dedos:
        if coordenadas[x][1] < coordenadas[x - 2][1]:
            return False
        if coordenadas[x - 1][1] < coordenadas[x - 3][1]:
            return False
    if coordenadas[4][0] > coordenadas[2][0] or coordenadas[2][1] > coordenadas[1][1]:
        return False
    cv2.putText(imgem, 'S', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_t(imgem, coordenadas):
    for x in dedos_g:
        if coordenadas[x][1] > coordenadas[x - 2][1]:
            return False
    if coordenadas[4][1] > coordenadas[8][1] and coordenadas[8][1] < coordenadas[6][1] and coordenadas[12][1] < \
            coordenadas[10][1]:
        return False
    if coordenadas[8][0] < coordenadas[4][0]:
        return False
    cv2.putText(imgem, 'T', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
    return True


def letra_u(imgem, coordenadas):
    if (coordenadas[8][0] - coordenadas[12][0]) > 30:
        return False
    if coordenadas[8][1] < coordenadas[6][1] and coordenadas[12][1] < coordenadas[10][1]:
        for x in [16, 20]:
            if coordenadas[x][1] < coordenadas[x - 2][1]:
                return False
        cv2.putText(imgem, 'U', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
        return True
    return False


def letra_v(imgem, coordenadas):
    if coordenadas[8][1] < coordenadas[6][1] and coordenadas[12][1] < coordenadas[10][1]:
        if abs(coordenadas[8][0] - coordenadas[12][0]) > 30:
            cv2.putText(imgem, 'V', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
            return True
    return False


video = cv2.VideoCapture(0)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    pontos = []

    if handsPoints:
        for points in handsPoints:
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)  # desenha os pontos e linhas das mãos
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                cv2.putText(img, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                            2)  # coloca os números das mãos
                pontos.append((cx, cy))

            if points:
                letra_a(img, pontos)
                letra_b(img, pontos)
                letra_c(img, pontos)
                letra_d(img, pontos)
                letra_e(img, pontos)
                letra_f(img, pontos)
                letra_g(img, pontos)
                letra_i(img, pontos)
                letra_l(img, pontos)
                letra_m(img, pontos)
                letra_n(img, pontos)
                letra_o(img, pontos)
                letra_p(img, pontos)
                letra_q(img, pontos)
                letra_r(img, pontos)
                letra_s(img, pontos)
                letra_t(img, pontos)
                letra_u(img, pontos)
                letra_v(img, pontos)

    cv2.imshow("Tradutor de LIBRAS", img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

video.release()
cv2.destroyAllWindows()