def getMotorSpeeds(currentX, currentY, targetX, targetY, pGain):
	from numpy import rad2deg, arctan, sqrt
	dx = targetX - currentX
	dy = targetY - currentY
	angle = rad2deg(arctan(dx/dy))
	r = 0
	l = 0
	v_x = dx*pGain
	v_y = dy*pGain
	v_mag = sqrt(v_x^2 + v_y^2)
	if angle > 0:
		if angle > 90:
			r = 0
			l = v_mag
		else:
			r = v_mag - v_x
			l = v_mag
	elif angle < 0:
		if angle < -90:
			l = 0
			r = v_mag
		else:
			l = v_mag + v_x
			r = v_mag
	else:
		l = v_mag
		r = v_mag
	return [r, l]