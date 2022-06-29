using Oceananigans
using NCDatasets
using Plots
using Printf


grid = RectilinearGrid(
  CPU(),
  size=(64, 64, 64), halo=(3, 3, 3),
  x = (0, 2),
  y = (0, 2),
  z = (0, 2),
  topology=(Periodic, Bounded, Periodic)
)
                   
x, y, z = nodes((Face, Center, Center), grid)

kwargs1 = (
  xlabel = "y",
  ylabel = "z",
  aspect = 1,
  fill = true,
  levels = 20,
  linewidth = 0,
  color = :balance,
  colorbar = true,
  #ylim = (0, 1),
  #xlim = (0, 1),
  titlefont = font(8),
  xtickfont = font(8),
  ytickfont = font(8),
  xguidefontsize = 8,
  yguidefontsize = 8
)

ds = NCDataset("./data/inertial_instability.nc", "r")

print(ds["u"])

anim = @animate for (iter, t) in enumerate(ds["time"])
    ω = ds["u"][32, :, :, iter]

    plot = contour(y, z, ω',
      title = @sprintf("u, t = %.1f", t); kwargs1...
    )

end

close(ds)

mp4(anim, "./plots/inertial_instability.mp4", fps=15)