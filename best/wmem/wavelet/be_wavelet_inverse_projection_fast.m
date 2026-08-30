function inv_proj = be_wavelet_inverse_projection_fast(obj,OPTIONS)
%BE_WAVELET_INVERSE_PROJECTION Compute the inverse projection from box to
%time courses

    nbSmp       = size(obj.ImageGridAmp,2);
    nbSmpTime   = size(obj.data,2) ;

    all_scales  = OPTIONS.automatic.selected_samples(2, :);
    all_transls = OPTIONS.automatic.selected_samples(3, :);

    unique_scales = unique(all_scales);
    inv_proj = sparse(nbSmp, nbSmpTime);

    for iScale = 1:length(unique_scales)
        
        iBoxes = find(all_scales == unique_scales(iScale));

        scales = all_scales(iBoxes);
        transls = all_transls(iBoxes);


        x = 1;
        y = nbSmpTime ./ (2.^scales(1)) + transls(1);
        wav = sparse(x, y, 1, 1, nbSmpTime);
        inv_wavelet    =   be_wavelet_inverse( wav, OPTIONS );

            
        nCols = size(inv_wavelet, 2);
        shiffting = 2.^scales(1);
        shiftAmounts = mod(shiffting * (transls(1) - transls(:)), size(inv_wavelet, 2));
        
        indices = mod(bsxfun(@plus, (1:nCols)-1, shiftAmounts), nCols) + 1;
        inv_proj(iBoxes, :) = inv_wavelet(indices);
    end
    inv_proj    =   inv_proj(:,obj.info_extension.start:obj.info_extension.end);

end

