def override_model_attrs(model, overrides):
  if overrides is not None and len(overrides.strip()):
    overrides = [p.split('=') for p in overrides.split(',')]
    for key, val in overrides:
      val_type = type(getattr(model, key))
      if val_type == bool:
        setattr(model, key, val in ['True', 'true', 't', '1'])
      elif val_type == list:
        setattr(model, key, val.split(';'))
      else:
        setattr(model, key, val_type(val))

  attrs = sorted([x for x in dir(model) if (not x.startswith('_') and not callable(getattr(model, x)))])
  
  summary = '\n'.join([
      '{},{}'.format(k, getattr(model, k))
      for k in attrs
  ])

  return model, summary
