import deeptrack as dt
import numpy as np

def generate_standard(image_size=64, noise_value=1.5e-4, position=(0,0)):
    
    z_pos = 0
    # z_pos = lambda: np.random.uniform(-15,15)

    particle=dt.MieSphere(position=(position[0]+image_size//2,position[1]+image_size//2), radius=7e-8, refractive_index=1.4, z=z_pos, position_objective=np.array([0, 0, 0]))
    args=dt.Arguments(hccoeff=lambda: np.random.uniform(-100,100))
    # args=dt.Arguments(hccoeff=0)
    pupil=dt.HorizontalComa(coefficient=args.hccoeff)
    optics=dt.Brightfield(NA=1.0,working_distance=.2e-3,aberration=pupil,wavelength=660e-9,resolution=.15e-6,magnification=1,output_region=(0,0,image_size,image_size),return_field=True,illumination_angle=np.pi) 
    
    def phase_adder(ph):
        def inner(image):
            image=image-1
            image=image*np.exp(1j*ph)
            image=image+1
            return image
        return inner
    
    phadd=dt.Lambda(phase_adder,ph=lambda: np.random.uniform(0,2*np.pi))
    # phadd=dt.Lambda(phase_adder,ph = np.pi)
    s0=optics(particle) #Apply the optics to the particle
    sample=s0>>phadd #Randomly change the relative phase of scattered light and reference light
    sample=(sample>>dt.ComplexGaussian(sigma=noise_value)) #Add noise to the images
    # sample=(sample>>dt.NormalizeMinMax(minv,maxv))
    im = sample.update()()
    positions = im.get_property('position', get_one=False)
    return im, positions


def generate_particle(image_size=64,
                    refractive_index_range = np.array([1.45, 1.46]),
                    # z = lambda: np.random.uniform(-30,30)):
                    z = 0,
                    # ph = lambda: np.random.uniform(0,2*np.pi),
                    ph = 1/16 * np.pi,
                    noise_value=1e-4,
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
    # refractive_index_medium = 1.49
    )

    NA = 1.3
    #NA = 1.49 krashar
    working_distance = 0.2e-3
    wavelength = 520e-9
    resolution = 65e-9
    magnification = 1

    # # HC = lambda: dt.HorizontalComa(coefficient=lambda c1: c1, c1=0 + np.random.randn() * 0.5)
    # # VC = lambda: dt.VerticalComa(coefficient=lambda c2:c2, c2=0 + np.random.randn() * 0.5)

    # HC = lambda: dt.HorizontalComa(coefficient=0)
    # VC = lambda: dt.VerticalComa(coefficient=0)

    # def crop(pupil_radius):
    #     def inner(image):
    #         x = np.arange(image.shape[0]) - image.shape[0] / 2
    #         y = np.arange(image.shape[1]) - image.shape[1] / 2
    #         X, Y = np.meshgrid(x, y)
    #         image[X ** 2 + Y ** 2 > pupil_radius ** 2] = 0
    #         return image
    #     return inner
    # CROP = dt.Lambda(crop, pupil_radius=32)  

    # pupil = dt.Lambda(HC) >> dt.Lambda(VC) >> CROP

    optics=dt.Brightfield(
        NA=NA,
        working_distance=working_distance,
        aberration=dt.Astigmatism(coefficient=5),
        # abberation=pupil,
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
            direction_degrees = np.random.uniform(0,180) # changed
            warp_factor = np.random.uniform(0, 0.05) # changed
            
            x = np.linspace(-np.pi, np.pi, N)
            y = np.linspace(-np.pi, np.pi, N)

            # Convert direction to radians
            direction_radians = np.radians(direction_degrees)

            # Calculate displacement for both x and y with warping
            warped_x = x * np.cos(direction_radians) + warp_factor * np.sin(direction_radians * x)
            warped_y = y * np.sin(direction_radians) + warp_factor * np.sin(direction_radians * y)

            # Generate 2D sine wave using the warped coordinates
            sine2D = np.sin((warped_x[:, np.newaxis] + warped_y) * frequency)# 128.0 + (127.0 * np.sin((warped_x[:, np.newaxis] + warped_y) * frequency))
            # sine2D = sine2D / 255.0

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

    #normalize between -1 and 1
    # def rescale(image):
    #     return 2 * (image - np.min(image)) / np.ptp(image) - 1
        
    return rescale(sample.update()())