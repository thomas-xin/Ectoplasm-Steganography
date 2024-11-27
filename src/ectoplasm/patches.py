import typing
import cv2
import numpy as np
from qrdet import (
    BBOX_XYXY,
    PADDED_QUAD_XY,
    crop_qr,
)
from qreader import (
    _SHARPEN_KERNEL,
    DecodeQRResult,
    wrap,
    decodeQR,
    ZBarSymbol,
    Decoded,
)

CorrectionsType = typing.Literal["cropped_bbox", "corrected_perspective"]

def _correct_perspective(
    self, image: np.ndarray, padded_quad_xy: np.ndarray
) -> np.ndarray:
    # Define the width and height of the quadrilateral
    width1 = np.sqrt(
        ((padded_quad_xy[0][0] - padded_quad_xy[1][0]) ** 2)
        + ((padded_quad_xy[0][1] - padded_quad_xy[1][1]) ** 2)
    )
    width2 = np.sqrt(
        ((padded_quad_xy[2][0] - padded_quad_xy[3][0]) ** 2)
        + ((padded_quad_xy[2][1] - padded_quad_xy[3][1]) ** 2)
    )

    height1 = np.sqrt(
        ((padded_quad_xy[0][0] - padded_quad_xy[3][0]) ** 2)
        + ((padded_quad_xy[0][1] - padded_quad_xy[3][1]) ** 2)
    )
    height2 = np.sqrt(
        ((padded_quad_xy[1][0] - padded_quad_xy[2][0]) ** 2)
        + ((padded_quad_xy[1][1] - padded_quad_xy[2][1]) ** 2)
    )

    # Take the maximum width and height to ensure no information is lost
    max_width = max(int(width1), int(width2))
    max_height = max(int(height1), int(height2))
    N = max(max_width, max_height)

    # Create destination points for the perspective transform. This forms an N x N square
    dst_pts = np.array(
        [[0, 0], [N - 1, 0], [N - 1, N - 1], [0, N - 1]], dtype=np.float32
    )

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(padded_quad_xy, dst_pts)

    # Perform the perspective warp
    dst_img = cv2.warpPerspective(image, M, (N, N))

    return dst_img

def _threshold_and_blur_decodings(
        self,
        image: np.ndarray,
        blur_kernel_sizes: tuple[tuple[int, int], ...] = ((3, 3),),
    ) -> list[Decoded]:
    assert (
        2 <= len(image.shape) <= 3
    ), f"image must be 2D or 3D (HxW[xC]) (uint8). Got {image.shape}"
    # decodedQR = decodeQR(image=image, symbols=[ZBarSymbol.QRCODE])
    # if len(decodedQR) > 0:
    #     return decodedQR

    # Try to binarize the image (Only works with 2D images)
    if len(image.shape) == 2:
        _, binary_image = cv2.threshold(
            image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        decodedQR = decodeQR(image=binary_image, symbols=[ZBarSymbol.QRCODE])
        if len(decodedQR) > 0:
            return decodedQR

    for kernel_size in blur_kernel_sizes:
        assert (
            isinstance(kernel_size, tuple) and len(kernel_size) == 2
        ), f"kernel_size must be a tuple of 2 elements. Got {kernel_size}"
        assert all(
            kernel_size[i] % 2 == 1 for i in range(2)
        ), f"kernel_size must be a tuple of odd elements. Got {kernel_size}"

        # If it not works, try to parse to sharpened grayscale
        blur_image = cv2.GaussianBlur(src=image, ksize=kernel_size, sigmaX=0)
        decodedQR = decodeQR(image=blur_image, symbols=[ZBarSymbol.QRCODE])
        if len(decodedQR) > 0:
            return decodedQR
    return []

def _decode_qr_zbar(
    self,
    image: np.ndarray,
    detection_result: dict[
        str, np.ndarray | float | tuple[float | int, float | int]
    ],
    scale_factors=None,
) -> list[DecodeQRResult]:
    cropped_bbox, _ = crop_qr(
        image=image, detection=detection_result, crop_key=BBOX_XYXY
    )
    cropped_quad, updated_detection = crop_qr(
        image=image, detection=detection_result, crop_key=PADDED_QUAD_XY
    )
    corrected_perspective = _correct_perspective(
        self,
        image=cropped_quad, padded_quad_xy=updated_detection[PADDED_QUAD_XY]
    )

    corrections = {
        "cropped_bbox": cropped_bbox,
        "corrected_perspective": corrected_perspective,
    }

    scale_factors = scale_factors or (1, 0.5, 2, 0.25, 3, 4)
    for scale_factor in scale_factors:
        for label, image in corrections.items():
            # If rescaled_image will be larger than 1024px, skip it
            # TODO: Decide a minimum size for the QRs based on the resize benchmark
            if (
                not all(25 < axis < 1024 for axis in image.shape[:2])
                and scale_factor != 1
            ):
                continue

            rescaled_image = cv2.resize(
                src=image,
                dsize=None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LANCZOS4,
            )
            decodedQR = decodeQR(image=rescaled_image, symbols=[ZBarSymbol.QRCODE])
            if len(decodedQR) > 0:
                return wrap(
                    scale_factor=scale_factor,
                    corrections=typing.cast(CorrectionsType, label),
                    flavor="original",
                    blur_kernel_sizes=None,
                    image=rescaled_image,
                    results=decodedQR,
                )
            if np.sum(rescaled_image) < np.prod(rescaled_image.shape).item() * 160:
                # For QRs with black background and white foreground, try to invert the image
                inverted_image = np.array(255) - rescaled_image
                decodedQR = decodeQR(inverted_image, symbols=[ZBarSymbol.QRCODE])
                if len(decodedQR) > 0:
                    return wrap(
                        scale_factor=scale_factor,
                        corrections=typing.cast(CorrectionsType, label),
                        flavor="inverted",
                        blur_kernel_sizes=None,
                        image=inverted_image,
                        results=decodedQR,
                    )

            # If it not works, try to parse to grayscale (if it is not already)
            if len(rescaled_image.shape) == 3:
                assert (
                    rescaled_image.shape[2] == 3
                ), f"Image must be RGB or BGR, but it has {image.shape[2]} channels."
                gray = cv2.cvtColor(rescaled_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = rescaled_image
            decodedQR = _threshold_and_blur_decodings(
                self,
                image=gray, blur_kernel_sizes=((5, 5), (7, 7))
            )
            if len(decodedQR) > 0:
                return wrap(
                    scale_factor=scale_factor,
                    corrections=typing.cast(CorrectionsType, label),
                    flavor="grayscale",
                    blur_kernel_sizes=((5, 5), (7, 7)),
                    image=gray,
                    results=decodedQR,
                )

            if len(rescaled_image.shape) == 3:
                # If it not works, try to sharpen the image
                sharpened_gray = cv2.cvtColor(
                    cv2.filter2D(
                        src=rescaled_image, ddepth=-1, kernel=_SHARPEN_KERNEL
                    ),
                    cv2.COLOR_RGB2GRAY,
                )
            else:
                sharpened_gray = cv2.filter2D(
                    src=rescaled_image, ddepth=-1, kernel=_SHARPEN_KERNEL
                )
            decodedQR = _threshold_and_blur_decodings(
                self,
                image=sharpened_gray, blur_kernel_sizes=((3, 3),)
            )
            if len(decodedQR) > 0:
                return wrap(
                    scale_factor=scale_factor,
                    corrections=typing.cast(CorrectionsType, label),
                    flavor="grayscale",
                    blur_kernel_sizes=((3, 3),),
                    image=sharpened_gray,
                    results=decodedQR,
                )

    return []