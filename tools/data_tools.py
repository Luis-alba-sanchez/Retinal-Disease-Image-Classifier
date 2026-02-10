


def crop_image_tensor(img_tensor):
    _, w, h = img_tensor.shape
    # Si l'image est en haute résolution (4288x2848)
    if h == 4288 and w == 2848:
        # Rogner à gauche de 260 pixels et à droite de 600 pixels
        return img_tensor[:, :, 260:-600]
    # Si l'image est en résolution standard (2144x1424)
    elif h == 2144 and w == 1424:
        # Rogner normalement (360 pixels de chaque côté)
        return img_tensor[:, :, 360:-360]
    elif h == 2048 and w == 1536:
        # Rogner à gauche de 260 pixels et à droite de 600 pixels
        return img_tensor[:, :, 280:-280]
    else:
        # Si la taille ne correspond à aucun cas connu, retourner l'image non rognée
        print(f"Taille inattendue : {h}x{w}")
        return img_tensor
