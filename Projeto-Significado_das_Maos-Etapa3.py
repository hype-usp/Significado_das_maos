import cv2
import mediapipe as mp

todos_dedos = [4, 8, 12, 16, 20]
pontos_ant = []  # Variável que armazena as coordenadas dos últimos 50 frames


# Gesto inicial
def letra_h1(coordenadas):
    if coordenadas[16][1] < coordenadas[14][1] or coordenadas[20][1] < coordenadas[18][1] or coordenadas[12][1] < \
            coordenadas[8][1]:
        return False
    if coordenadas[12][1] > coordenadas[10][1] or coordenadas[8][1] > coordenadas[6][1] or coordenadas[4][1] > \
            coordenadas[2][1]:
        return False
    if coordenadas[12][0] < coordenadas[4][0] < coordenadas[8][0]:
        return True


# Gesto final
def letra_h2(coordenadas):
    if coordenadas[16][1] < coordenadas[14][1] or coordenadas[20][1] < coordenadas[18][1] or coordenadas[12][1] < \
            coordenadas[8][1]:
        return False
    if coordenadas[12][1] > coordenadas[10][1] or coordenadas[8][1] > coordenadas[6][1] or coordenadas[4][1] > \
            coordenadas[2][1]:
        return False
    if coordenadas[12][0] < coordenadas[10][0]:
        return False
    if coordenadas[12][0] > coordenadas[4][0] > coordenadas[8][0]:
        return True


# Função pra identificar a letra H em libras
def letra_h(anteriores):
    h1 = False
    h2 = False
    for i in range(0, len(anteriores)):  # loop pra passar em todos os frames armazenados
        if letra_h1(anteriores[i]):
            h1 = True
        if h1:
            if letra_h2(anteriores[i]):
                h2 = True
    if h2:
        return True


def letra_i(coordenadas):
    if coordenadas[20][1] > coordenadas[19][1] or coordenadas[0][1] < coordenadas[13][1]:
        return False
    if coordenadas[4][0] > coordenadas[3][0]:
        return False
    for x in todos_dedos[1:4]:
        if coordenadas[x][1] < coordenadas[x - 2][1]:
            return False
    return True


def letra_j2(coordenadas):
    if coordenadas[5][1] > coordenadas[17][1]:
        return False
    if coordenadas[6][1] > coordenadas[7][1] or coordenadas[10][1] > coordenadas[11][1] or coordenadas[14][1] > \
            coordenadas[15][1]:
        return False
    if coordenadas[20][0] < coordenadas[18][0]:
        return False
    return True


def letra_j(anteriores):
    j1 = False
    j2 = False
    for i in range(0, len(anteriores)):  # loop pra passar em todos os frames armazenados
        if letra_i(anteriores[i]):
            j1 = True
        if j1:
            if letra_j2(anteriores[i]):
                j2 = True
    if j2:
        return True


def letra_k1(coordenadas):
    if coordenadas[2][1] > coordenadas[0][1] or coordenadas[10][1] > coordenadas[12][1] or coordenadas[12][1] > \
            coordenadas[4][1]:
        return False
    if coordenadas[6][0] < coordenadas[7][0] < coordenadas[8][0] or coordenadas[12][0] > coordenadas[11][0]:
        return False
    if coordenadas[3][1] > coordenadas[9][1]:
        return True


def letra_k2(coordenadas):
    if coordenadas[2][1] > coordenadas[0][1] or coordenadas[10][1] < coordenadas[12][1] or coordenadas[15][1] > \
            coordenadas[16][1]:
        return False
    if coordenadas[3][1] < coordenadas[9][1]:
        return True


def letra_k(anteriores):
    k1 = False
    k2 = False
    for i in range(0, len(anteriores)):  # loop pra passar em todos os frames armazenados
        if letra_k1(anteriores[i]):
            k1 = True
        if k1:
            if letra_k2(anteriores[i]):
                k2 = True
    if k2:
        return True


def letra_x1(coordenadas):
    for x in todos_dedos[2:]:
        if coordenadas[x][1] < coordenadas[x - 2][1]:
            return False
    if coordenadas[0][1] < coordenadas[1][1] or coordenadas[0][1] < coordenadas[20][1] or coordenadas[0][1] < \
            coordenadas[16][1] or coordenadas[6][1] < coordenadas[7][1] or coordenadas[4][1] > coordenadas[12][1]:
        return False
    if coordenadas[4][0] > coordenadas[3][0] or coordenadas[0][0] > coordenadas[1][0] or coordenadas[10][0] > \
            coordenadas[6][0]:
        return False
    return True


def letra_x2(coordenadas):
    if coordenadas[0][0] > 320:
        return True


def letra_x(anteriores):
    x1 = False
    x2 = False
    for i in range(0, len(anteriores)):  # loop pra passar em todos os frames armazenados
        if letra_x1(anteriores[i]):
            x1 = True
        if x1:
            if letra_x2(anteriores[i]):
                x2 = True
    if x2:
        return True


def letra_y1(coordenadas):
    if coordenadas[20][1] < coordenadas[19][1] or coordenadas[20][0] < coordenadas[19][0]:
        return False
    if coordenadas[4][1] < coordenadas[3][1] or coordenadas[4][0] < coordenadas[3][0]:
        return False
    if coordenadas[20][1] < coordenadas[13][1]:
        return False
    return True


def letra_y2(coordenadas):
    if coordenadas[20][1] < coordenadas[19][1] or coordenadas[20][0] < coordenadas[18][0]:
        return False
    if coordenadas[4][1] < coordenadas[3][1] or coordenadas[4][0] < coordenadas[3][0]:
        return False
    if coordenadas[20][1] > coordenadas[13][1]:
        return True


def letra_y(anteriores):
    y1 = False
    y2 = False
    for i in range(0, len(anteriores)):  # loop pra passar em todos os frames armazenados
        if letra_y1(anteriores[i]):
            y1 = True
        if y1:
            if letra_y2(anteriores[i]):
                y2 = True
    if y2:
        return True


def letra_z1(coordenadas):
    if coordenadas[8][1] < coordenadas[7][1]:
        return False
    if coordenadas[4][0] > coordenadas[3][0] or coordenadas[12][1] > coordenadas[11][1] or coordenadas[16][1] > \
            coordenadas[15][1] or coordenadas[20][1] > coordenadas[19][1]:
        return False

    return True


def letra_z2(coordenadas, punhotemp):
    if coordenadas[8][1] < coordenadas[7][1]:
        return False
    if coordenadas[0][0] > punhotemp[0][0]:
        return True


def letra_z3(coordenadas, punhotemp):
    if coordenadas[8][1] < coordenadas[7][1]:
        return False
    if coordenadas[0][0] < punhotemp[0][0] and coordenadas[0][1] < punhotemp[0][1]:
        return True


def letra_z4(coordenadas, punhotemp):
    if coordenadas[8][1] < coordenadas[7][1]:
        return False
    if coordenadas[0][0] > punhotemp[0][0]:
        return True


def letra_z(anteriores):
    z1 = False
    z2 = False
    z3 = False
    z4 = False
    for i in range(0, len(anteriores)):  # loop pra passar em todos os frames armazenados
        if letra_z1(anteriores[i]):
            punhotemp = anteriores[i]
            z1 = True
            if z1:
                if letra_z2(anteriores[i], punhotemp):
                    punhotemp = anteriores[i]
                    z2 = True

                if z2:
                    if letra_z3(anteriores[i], punhotemp):
                        punhotemp = anteriores[i]
                        z3 = True

                    if z3:
                        if letra_z4(anteriores[i], punhotemp):
                            z4 = True
        if z4:
            return True


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
                # Para identificar a letra H
                if letra_h(pontos_ant):
                    cv2.putText(img, 'H', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)

                if letra_i(pontos):
                    cv2.putText(img, 'I', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)

                if letra_j(pontos_ant):
                    cv2.putText(img, 'J', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)

                if letra_k(pontos_ant):
                    cv2.putText(img, 'K', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)

                if letra_x(pontos_ant):
                    cv2.putText(img, 'X', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)

                if letra_y(pontos_ant):
                    cv2.putText(img, 'Y', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)

                if letra_z(pontos_ant):
                    cv2.putText(img, 'Z', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)

            # Anexando as coordenadas das mãos na variável
            pontos_ant.append(pontos)
            # Limitar o número de posições para evitar crescimento infinito
            if len(pontos_ant) > 50:
                pontos_ant.pop(0)

    cv2.imshow("Tradutor de LIBRAS", img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

video.release()
cv2.destroyAllWindows()