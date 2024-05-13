import deeptrack as dt
import numpy as np

def generate_particle(image_size=64,
                    refractive_index_range = np.array([1.45, 1.46]),
                    z = 0,
                    ph = 1/16 * np.pi,
                    noise_value=1.5e-4,
                    position=(0,0),
                    polarization_angle = lambda: np.random.rand() * 2 * np.pi,
                    radius = lambda: np.random.uniform(45e-9, 55e-9),
                    ):
    
    particle=dt.MieSphere(
    position=(image_size//2 + position[0], image_size//2 + position[1]),
    radius=radius,
    refractive_index=lambda: np.random.uniform(refractive_index_range[0], refractive_index_range[1]), 
    z=z,
    position_objective=(np.random.uniform(-250,250)*1e-6,np.random.uniform(-250,250)*1e-6, np.random.uniform(-15,15)*1e-6),
    position_unit="pixel",
    refractive_index_medium=1.33,
    )

    NA = 1.3
    #NA = 1.49 krashar
    working_distance = 0.2e-3
    wavelength = 520e-9
    resolution = 65e-9
    magnification = 1

    optics=dt.Brightfield(
        NA=NA,
        working_distance=working_distance,
        aberration=dt.Astigmatism(coefficient=5),
        wavelength=wavelength,
        resolution=resolution,
        magnification=magnification,
        output_region=(0,0,image_size,image_size),
        padding=(image_size//2,) * 4,
        polarization_angle=polarization_angle,
        return_field=True,
        backscatter=True,
        illumination_angle=np.pi,
        )
    
    def phase_adder(ph):
        def inner(image):
            image=image-1
            image=image*np.exp(1j*ph)
            image=image+1
            return np.abs(image)
        return inner
    
    def generate_sine_wave_2D(p):

        def inner(image):
            N = image.shape[0]
            frequency = np.random.uniform(1, 10)
            direction_degrees = np.random.uniform(0,180)
            warp_factor = np.random.uniform(0, 0.05)
            
            x = np.linspace(-np.pi, np.pi, N)
            y = np.linspace(-np.pi, np.pi, N)

            # Convert direction to radians
            direction_radians = np.radians(direction_degrees)

            # Calculate displacement for both x and y with warping
            warped_x = x * np.cos(direction_radians) + warp_factor * np.sin(direction_radians * x)
            warped_y = y * np.sin(direction_radians) + warp_factor * np.sin(direction_radians * y)

            # Generate 2D sine wave using the warped coordinates
            sine2D = np.sin((warped_x[:, np.newaxis] + warped_y) * frequency)

            #flip or mirror the pattern
            if np.random.rand()>0.5:
                sine2D=np.flip(sine2D,0)
            if np.random.rand()>0.5:
                sine2D=np.flip(sine2D,1)
            if np.random.rand()>0.5:
                sine2D=np.transpose(sine2D)

            image = image + np.expand_dims(sine2D, axis = -1)*p

            return image

        return inner
    
    import skimage.measure
    def ds(factor=2):
        def inner(image):
            return skimage.measure.block_reduce(image, (factor, factor, 1), np.mean)
        return inner

    phadd = dt.Lambda(phase_adder, ph=ph)
    wave = dt.Lambda(generate_sine_wave_2D, p=lambda: np.random.uniform(0.0001,0.00015))

    s0=optics(particle)
    sample=s0  >>phadd
    sample = sample  >> wave >> dt.Gaussian(sigma=noise_value) 

    def rescale(image):
        correction = 0.004
        return 2* (image - (1 - correction)) / (1 + correction - (1 - correction)) - 1
        
    return rescale(sample.update()())