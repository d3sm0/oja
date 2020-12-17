### linearized graph

**fwd mode**
```
J = [
        [x1 * e + a*c*d*f, b*e + b*c*d*f],
        [a*c*d*g, b*c*d*g]
        ]
```
**Cross-country**
```
r = cd
s = rf 
t = rg
u = e + s
J = [ [au, bu], [at, bt] ]
```

Goal:

```
min sum_{i, j} f[i, j]
```

### asserts:

```
a_ij = e_i A e_j = c_ij
c_ij =  d_j phi_i / d_j v_j

```
